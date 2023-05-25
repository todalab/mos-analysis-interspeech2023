import argparse
import csv
from preprocess import VoiceMOSChallenge2022
from tail_probability import Hoeffding, BernoulliChernoff, BernoulliExactAsymptotics, CentralLimitTheorem, CentralLimitTheoremT


def denormalize(v):
    return v * 4 + 1


def denormalize_interval(v):
    return v * 4


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='input file path')
    parser.add_argument('-d', '--delta', type=float, default=0.05,
                        help='confidence level')
    parser.add_argument('-o', '--output1', type=str,
                        help='Confidence interval output file path', required=True)
    parser.add_argument('-p', '--output2', type=str,
                        help='The number of insignificant systems output file path', required=True)

    args = parser.parse_args()
    delta = args.delta
    out_file1 = args.output1
    out_file2 = args.output2

    preprocesssor = VoiceMOSChallenge2022()

    environment, arms = preprocesssor.preprocess(args.input)

    hoeffding = Hoeffding(delta)
    bernoulliChernoff = BernoulliChernoff(delta)
    bernoulliExactAsymptotics = BernoulliExactAsymptotics(delta)
    centralLimitTheorem = CentralLimitTheorem(delta)
    centralLimitTheoremT = CentralLimitTheoremT(delta)

    stats = environment.sample_stats()
    ms = []
    interval_hoeffdings = []
    interval_bernoulliChernoffs = []
    interval_bernoulliExactAsymptoticss = []
    interval_centralLimitTheorems = []
    interval_centralLimitTheoremTs = []
    with open(out_file1, 'w', newline='') as csvfile:
        ci_writer = csv.writer(csvfile, delimiter=',')
        for system, (sample_mean, sample_stddev, n_samples) in stats.items():
            ms.append(sample_mean)
            interval_hoeffding = hoeffding.confidence_interval(n_samples)
            interval_bernoulliChernoff = bernoulliChernoff.confidence_interval(n_samples, sample_mean)
            interval_bernoulliExactAsymptotics = bernoulliExactAsymptotics.confidence_interval(n_samples, sample_mean)
            interval_centralLimitTheorem = centralLimitTheorem.confidence_interval(n_samples, sample_stddev)
            interval_centralLimitTheoremT = centralLimitTheoremT.confidence_interval(n_samples, sample_stddev)

            interval_hoeffdings.append(interval_hoeffding)
            interval_bernoulliChernoffs.append(interval_bernoulliChernoff)
            interval_bernoulliExactAsymptoticss.append(interval_bernoulliExactAsymptotics)
            interval_centralLimitTheorems.append(interval_centralLimitTheorem)
            interval_centralLimitTheoremTs.append(interval_centralLimitTheoremT)

            # print(system,
            #       f"{sample_mean:.3f}",
            #       f"{interval_hoeffding:.3f}",
            #       f"{interval_bernoulliChernoff:.3f}",
            #       f"{interval_bernoulliExactAsymptotics:.3f}",
            #       f"{interval_centralLimitTheorem:.3f}",
            #       f"{interval_centralLimitTheoremT:.3f}")
            # print(','.join([system,
            #       f"{denormalize(sample_mean):.2f}",
            #       f"{denormalize_interval(interval_hoeffding):.2f}",
            #       f"{denormalize_interval(interval_bernoulliChernoff):.2f}",
            #       f"{denormalize_interval(interval_bernoulliExactAsymptotics):.2f}",
            #       f"{denormalize_interval(interval_centralLimitTheorem):.2f}",
            #       f"{denormalize_interval(interval_centralLimitTheoremT):.2f}"]))
            ci_writer.writerow([system,
                               f"{sample_mean:.3f}",
                               f"{interval_hoeffding:.3f}",
                               f"{interval_bernoulliChernoff:.3f}",
                               f"{interval_bernoulliExactAsymptotics:.3f}",
                               f"{interval_centralLimitTheorem:.3f}",
                               f"{interval_centralLimitTheoremT:.3f}"])


    res = [[], [], [], [], []]

    for r, iis in zip(res, [interval_hoeffdings, interval_bernoulliChernoffs, interval_bernoulliExactAsymptoticss, interval_centralLimitTheorems, interval_centralLimitTheoremTs]):
        for i, (mm, ci) in enumerate(zip(ms, iis)):
            ls = mm - ci
            us = mm + ci
            n = 0
            for j, (mt, ci) in enumerate(zip(ms, iis)):
                ut = mt + ci
                lt = mt - ci
                is_covered = ls <= ut if mm > mt else lt <= us
                if i != j and is_covered:
                    n += 1
            r.append(n)

    with open(out_file2, 'w', newline='') as csvfile:
        ni_writer = csv.writer(csvfile, delimiter=',')
        for m, n_Hoeffding, n_Chernoff, n_ExactAsymptotics, n_centralLimitTheorems, n_centralLimitTheoremT in zip(ms, res[0], res[1], res[2], res[3], res[4]):
            # print(','.join([str(m),
            #                 str(n_Hoeffding),
            #                 str(n_Chernoff),
            #                 str(n_ExactAsymptotics),
            #                 str(n_centralLimitTheorems),
            #                 str(n_centralLimitTheoremT)]))
            ni_writer.writerow([str(m),
                               str(n_Hoeffding),
                               str(n_Chernoff),
                               str(n_ExactAsymptotics),
                               str(n_centralLimitTheorems),
                               str(n_centralLimitTheoremT)])

    # print(max(res[0]), min(res[0]), sum(res[0])/len(res[0]), sorted(res[0])[len(res[0])//2])
    # print(max(res[1]), min(res[1]), sum(res[1])/len(res[1]), sorted(res[1])[len(res[1])//2])
    # print(max(res[2]), min(res[2]), sum(res[2])/len(res[2]), sorted(res[2])[len(res[2])//2])
    # print(max(res[3]), min(res[3]), sum(res[3])/len(res[3]), sorted(res[3])[len(res[3])//2])
