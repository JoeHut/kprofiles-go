// Package kprofiles provides an implementation of K-Profiles, a nonlinear
// clustering method for pattern detection in high-dimensional data, which was
// described by Wang et al.
package kprofiles

import (
	"bytes"
	"fmt"
	"math"
	"math/rand"
	"strconv"
	"strings"
	"time"

	"github.com/gonum/matrix/mat64"
	"github.com/senseyeio/roger"
)

// rClient connects to an R Server for calculating the shortest Hamiltonian
// Path when the clusters are reordered
var rClient roger.RClient

// initialization of the connection to the R Server. The function panics if not
// possiblei and expects the server to run on the default port on localhost.
// Also random is seeded.
func init() {
	var err error
	rClient, err = roger.NewRClient("127.0.0.1", 6311)
	if err != nil {
		panic("Connection to R Server could not be established")
	}
	rand.Seed(int64(time.Now().Nanosecond()))
}

// Cluster stores information about the contents of a cluster
type Cluster struct {
	// ColumnOrder stores the indices of the actual matrix columns in the order
	// of the current cluster
	ColumnOrder []int

	// Rows holds the indices of the actual matrix that are currently clustered
	Rows []int

	// current distance of the Hamiltonian Path based on the Manhattan Distance
	Distance float64

	// the last calculated distance of the Hamiltonian Path
	lastDistance float64
}

type Kprofiles struct {
	matrix         *mat64.Dense // base matrix
	nullDistMean   []float64    // Means of all rows under the null distribution
	nullDistStdDev []float64    // Standard Deviations of all rows under the null distribution
	Clusters       []*Cluster
}

// NewKprofiles returns a new object with the provided matrix m and the needed
// Cluster objects based on clusterCount
func NewKprofiles(m *mat64.Dense, clusterCount int) (*Kprofiles, error) {
	if m == nil {
		return nil, fmt.Errorf("Provided matrix is nil")
	}
	if clusterCount < 2 {
		return nil, fmt.Errorf("There have to be at least 2 clusters")
	}

	rows, columns := m.Dims()
	clusters := make([]*Cluster, clusterCount)

	for i := 0; i < clusterCount; i++ {
		clusters[i] = &Cluster{
			ColumnOrder:  make([]int, columns),
			Rows:         make([]int, rows),
			Distance:     math.MaxFloat64,
			lastDistance: 0,
		}
		clusters[i].initialize()
	}

	return &Kprofiles{
		matrix:         m,
		nullDistMean:   make([]float64, rows),
		nullDistStdDev: make([]float64, rows),
		Clusters:       clusters,
	}, nil
}

// NullDistParameters calculates the mean and standard deviation of the dcol
// for each row under the null distribution that there is no dependence
// regarding the order of the values inside a row of the matrix. Each row is
// permuttated p times and the dcol is calculated each time. The results are
// stored in the according arrays of k. Returns an error if one of the
// statistical computations fails.
func (k *Kprofiles) NullDistParameters(p int) error {

	rows, _ := k.matrix.Dims()

	for i := 0; i < rows; i++ {
		// copy the row
		row := k.matrix.RawRowView(i)
		rowCopy := make([]float64, len(row))
		copy(rowCopy, row)

		permutatedDcols := make([]float64, p)

		for j := 0; j < p; j++ {
			// shuffeling the row
			for element := range rowCopy {
				k := rand.Intn(element + 1)
				rowCopy[element], rowCopy[k] = rowCopy[k], rowCopy[element]
			}
			dcol, err := dcol(rowCopy)
			if err != nil {
				return err
			}
			permutatedDcols[j] = dcol
		}

		var err error
		k.nullDistMean[i], err = mean(permutatedDcols)
		if err != nil {
			return err
		}

		k.nullDistStdDev[i], err = stdDev(permutatedDcols, k.nullDistMean[i])
		if err != nil {
			return err
		}
	}
	return nil
}

// Cluster is the core algorithm of K-Profiles. It clusters the rows in its
// base matrix dependent on the order of the columns and the resulting dcol
// values. A row is attached to a cluster where its p-value is minimal and less
// than the current p-value cutoff alpha. Alpha is initialized with sAlpha and
// then in each iteration decreased until it reaches gAlpha. The clustering is
// complete if each cluster is stable or r iterations are done.
func (k *Kprofiles) Cluster(sAlpha, gAlpha float64, r int) error {
	if sAlpha < gAlpha {
		return fmt.Errorf("The start value of alpha is smaller than the goal value")
	}
	alpha := sAlpha
	for rounds := 0; !clustersStable(k.Clusters) && rounds < r; rounds++ {
		rows, _ := k.matrix.Dims()
		k.resetClusters()
		for i := 0; i < rows; i++ {
			row := k.matrix.RawRowView(i)
			var bestCluster *Cluster
			bestPvalue := math.MaxFloat64
			for _, c := range k.Clusters {
				clusterRow, err := getOrderedRow(row, c.ColumnOrder)
				if err != nil {
					return err
				}
				dcol, err := dcol(clusterRow)
				if err != nil {
					return err
				}
				pvalue := pvalue(dcol, k.nullDistMean[i], k.nullDistStdDev[i])

				// check if the calculated p-value is statistically
				// signinficant and store the current cluster as the best if
				// there hasn't been a better p-value
				if pvalue <= alpha && pvalue <= bestPvalue {
					bestPvalue = pvalue
					bestCluster = c
				}
			}

			// attach the row to the best fitting cluster regarding the p-value
			// if it is statistically signinficant
			if bestCluster != nil {
				bestCluster.Rows = append(bestCluster.Rows, i)
			}
		}

		var err error
		alpha, err = sidak(alpha, gAlpha, len(k.Clusters))
		if err != nil {
			return err
		}

		for _, c := range k.Clusters {
			if len(c.Rows) > 0 {
				err = c.reorder(k.matrix)
				if err != nil {
					return err
				}
			}
		}
	}
	return nil
}

func (k *Kprofiles) resetClusters() {
	for _, c := range k.Clusters {
		c.resetRows()
	}
}

// reorder uses the R TSP library to reorder the columns based on the currently
// clustered rows.
func (c *Cluster) reorder(matrix *mat64.Dense) error {
	var value interface{}
	var err error
	session, err := rClient.GetSession()
	if err != nil {
		return fmt.Errorf("R: Session could not be established: " + err.Error())
	}
	defer session.Close()

	value, err = session.Eval("library(\"TSP\")")
	if err != nil {
		return fmt.Errorf("R: TSP library could not be loaded: " + err.Error())
	}

	var buffer bytes.Buffer

	// R matrices are column based, therefore the row- and column-counts
	// are exchanged so that we directly have the transposed.
	buffer.WriteString("A <- matrix(c(" + c.valuesAsString(matrix) + ")")
	buffer.WriteString(",nrow=" + strconv.Itoa(len(c.ColumnOrder)))
	buffer.WriteString(",ncol=" + strconv.Itoa(len(c.Rows)) + ");")
	buffer.WriteString("d<-dist(A,method=\"man\");")
	buffer.WriteString("tsp<-TSP(d);")
	buffer.WriteString("tsp<-insert_dummy(tsp, label=\"cut\");")
	buffer.WriteString("tour<-solve_TSP(tsp,method=\"nn\");")
	buffer.WriteString("path<-cut_tour(tour,\"cut\");")
	buffer.WriteString("path;")

	value, err = session.Eval(buffer.String())
	if err != nil {
		return fmt.Errorf("R: tsp could not be solved: " + err.Error())
	} else {
		if newOrder, ok := value.([]int32); ok {
			c.ColumnOrder = c.ColumnOrder[:0]
			for i := range newOrder {
				c.ColumnOrder = append(c.ColumnOrder, int(newOrder[i])-1)
			}
		} else {
			return fmt.Errorf("R: returned path is not an int array")
		}
	}

	value, err = session.Eval("attributes(tour)$tour_length")
	if err != nil {
		return fmt.Errorf("R: tour distance could not be get: " + err.Error())
	} else {
		if newDistance, ok := value.(float64); ok {
			c.lastDistance = c.Distance
			c.Distance = newDistance
		} else {
			return fmt.Errorf("R: returned distance is not a float64 value")
		}
	}
	return nil
}

// initialize sets a intial random order for each cluster in k and puts
// every row in every cluster for the initial p-value calculation
func (c *Cluster) initialize() {
	c.ColumnOrder = rand.Perm(len(c.ColumnOrder))
	for i, _ := range c.Rows {
		c.Rows[i] = i
	}
}

func (c *Cluster) resetRows() {
	c.Rows = c.Rows[:0]
}

// valuesAsString returns the values of a cluster as comma-seperated row-based
// string.
func (c *Cluster) valuesAsString(matrix *mat64.Dense) string {
	stringArray := make([]string, 0)
	for _, v := range c.Rows {
		for j := 0; j < len(c.ColumnOrder); j++ {
			stringArray = append(stringArray, strconv.FormatFloat(matrix.At(v, j), 'f', -1, 64))
		}
	}
	return strings.Join(stringArray, ",")
}

// getOrderedRow returns a copy of baseRow ordered by the mapping of order
func getOrderedRow(baseRow []float64, order []int) ([]float64, error) {
	if len(baseRow) != len(order) {
		return nil, fmt.Errorf("Row couldn't be ordered. baseRow has length %d, order array has length %d", len(baseRow), len(order))
	}
	orderedRow := make([]float64, len(baseRow))
	for i := 0; i < len(baseRow); i++ {
		orderedRow[i] = baseRow[order[i]]
	}
	return orderedRow, nil
}

// clustersStable reports if it is no longer possible to make further progress in the
// calculation of the clusters. Returns true, if all clusters are empty (no
// rows get attached to any cluster) or no cluster made progress regarding the
// Hamiltonian Length in the last cluster round.
func clustersStable(clusters []*Cluster) bool {
	allClustersEmpty := true
	noDistanceProgress := true
	for _, c := range clusters {
		if len(c.Rows) != 0 {
			allClustersEmpty = false
		}
		if c.Distance != c.lastDistance {
			noDistanceProgress = false
		}
	}
	return allClustersEmpty || noDistanceProgress
}
