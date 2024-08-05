import random
import math

random.seed(42)


def calculate_sample_size(
        population_size,
        margin_error=.05,
        confidence_level=.99,
        sigma=1 / 2
):
    """
    Calculate the minimal sample size to use to achieve a certain
    margin of error and confidence level for a sample estimate
    of the population mean.
    Inputs
    -------
    population_size: integer
        Total size of the population that the sample is to be drawn from.
    margin_error: number
        Maximum expected difference between the true population parameter,
        such as the mean, and the sample estimate.
    confidence_level: number in the interval (0, 1)
        If we were to draw a large number of equal-size samples
        from the population, the true population parameter
        should lie within this percentage
        of the intervals (sample_parameter - e, sample_parameter + e)
        where e is the margin_error.
    sigma: number
        The standard deviation of the population.  For the case
        of estimating a parameter in the interval [0, 1], sigma=1/2
        should be sufficient.
    """

    zdict = {
        .90: 1.645,
        .91: 1.695,
        .99: 2.576,
        .97: 2.17,
        .94: 1.881,
        .93: 1.812,
        .95: 1.96,
        .98: 2.326,
        .96: 2.054,
        .92: 1.751
    }
    if confidence_level in zdict:
        z = zdict[confidence_level]
    else:
        raise f"confidence_level {confidence_level} isn't pre-calculated"
    N = population_size
    M = margin_error
    numerator = z ** 2 * sigma ** 2 * (N / (N - 1))
    denom = M ** 2 + ((z ** 2 * sigma ** 2) / (N - 1))
    return numerator / denom


def nth_combination(iterable, r, index):
    "Equivalent to list(combinations(iterable, r))[index]"
    pool = tuple(iterable)
    n = len(pool)
    c = math.comb(n, r)
    if index < 0:
        index += c
    if index < 0 or index >= c:
        raise IndexError
    result = []
    while r:
        c, n, r = c * r // n, n - 1, r - 1
        while index >= c:
            index -= c
            c, n = c * (n - r) // n, n - 1
        result.append(pool[-1 - n])
    return tuple(result)


def powerset_with_sampling(seq, confidence_level=0.95):
    """
    Returns a random sample of subsets of all possible sizes from the set.
    """
    all_subsets = []

    for subset_size in range(1, len(seq) + 1):
        total_subsets = math.comb(len(seq), subset_size)
        if total_subsets <= 1:
            target_samples = total_subsets
        else:
            target_samples = int(
                math.ceil(calculate_sample_size(total_subsets, margin_error=0.08, confidence_level=confidence_level)))

        print(subset_size, total_subsets, target_samples)
        # Generate random indices without replacement
        sampled_indices = set()

        # Use a while loop to manually sample indices without replacement
        while len(sampled_indices) < target_samples:
            index = random.randint(0, total_subsets - 1)
            sampled_indices.add(index)

        sampled_subsets = [nth_combination(seq, subset_size, i) for i in sampled_indices]
        assert len(set(sampled_subsets)) == target_samples

        all_subsets.extend(sampled_subsets)

    return all_subsets


if __name__ == '__main__':
    num_classes = 100
    sequence = list(range(num_classes))
    confidence_level = 0.95

    all_sampled_subsets = powerset_with_sampling(sequence, confidence_level)
    all_ex = []

    with open("./target_classes.num_classes_100.sample.list", "w+") as out_f:
        for s in all_sampled_subsets:
            if len(s) > 1:
                out_f.write("{}\n".format(" ".join([str(e) for e in s])))
