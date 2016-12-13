package kprofiles

import (
	"fmt"
	"math"

	gauss "github.com/chobie/go-gaussian"
)

// dcol calculates the distance based on conditionally ordered lists. This is
// the mean of the differences between adjacent values in list
func dcol(list []float64) (float64, error) {
	if len(list) < 2 {
		return 0, fmt.Errorf("At least 2 values needed to calculate dcol")
	}

	first := list[0]
	var sum float64

	for i := 1; i < len(list); i++ {
		sum = sum + math.Abs(list[i]-first)
		first = list[i]
	}

	factor := float64(1) / float64(len(list)-1)
	dcol := factor * sum
	return dcol, nil
}

// mean returns the arithmetic mean of values
func mean(values []float64) (float64, error) {
	if len(values) == 0 {
		return 0, fmt.Errorf("The provided slice is empty")
	}
	var sum float64

	for _, v := range values {
		sum += v
	}

	return sum / float64(len(values)), nil
}

// variance returns the variance of values based on the provided mean
func variance(values []float64, mean float64) (float64, error) {
	if len(values) == 0 {
		return 0, fmt.Errorf("The provided slice is empty")
	}
	var sum float64

	for _, v := range values {
		sum += math.Pow(v-mean, 2)
	}

	return sum / float64(len(values)), nil
}

// stdDev returns the standard deviation of values based on the provided mean
func stdDev(values []float64, mean float64) (float64, error) {
	variance, err := variance(values, mean)
	if err != nil {
		return 0, err
	}

	return math.Sqrt(variance), nil
}

// pvalue calculates the left-sided p-value of a normal distribution with mean
// mean and standard deviation std. This is the integral over the probability
// density function from -∞ to observed, which is the same as the value of the
// cumulative distribution function at observed
func pvalue(observed, mean, std float64) float64 {
	dist := gauss.NewGaussian(mean, math.Pow(std, 2))
	return dist.Cdf(observed)
}

// recalculateAlpha implements the Sidak correction to counteract the multiple
// comparison problem. Returns the old alpha value if the minimum threshold is
// already reached.
func sidak(alpha, minAlpha float64, hypotheses int) (float64, error) {
	if hypotheses == 0 {
		return 0, fmt.Errorf("Alpha can't be calculated without any cluster")
	}
	if alpha > minAlpha {
		return 1 - math.Pow((1-alpha), (1/float64(hypotheses))), nil
	}

	return alpha, nil
}
