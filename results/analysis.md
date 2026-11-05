## Test accuracy (2000 tasks, greedy), outcome shares averaged over seeds

| kind           |   n_seeds |   acc_std |   correct |   div_zero |   invalid |   near |   no_answer |   wrong_numbers |
|:---------------|----------:|----------:|----------:|-----------:|----------:|-------:|------------:|----------------:|
| base           |         1 |   nan     |     0.116 |          0 |         0 |  0.509 |       0     |           0.374 |
| A binary       |         5 |     0.123 |     0.705 |          0 |         0 |  0.149 |       0.128 |           0.018 |
| B lenient      |         1 |   nan     |     0.452 |          0 |         0 |  0.435 |       0     |           0.114 |
| C partial      |         8 |     0.021 |     0.708 |          0 |         0 |  0.28  |       0     |           0.012 |
| C-shuf placebo |         8 |     0.125 |     0.59  |          0 |         0 |  0.403 |       0     |           0.008 | 

## Differences in test accuracy, 95% Welch CI over seeds

- C partial - C-shuf placebo: +11.8 pp, CI [+1.3, +22.2] -> positive
- C partial - A binary: +0.2 pp, CI [-15.0, +15.4] -> undetermined
- C-shuf placebo - A binary: -11.6 pp, CI [-27.6, +4.5] -> undetermined

Post-hoc paired-seed difference, C - C-shuf: CI [+1.53, +22.06] pp

C-shuf, among near answers: singletons 2.3%; identical rewards in group 37.9%; assigned reward unchanged 69.4%

## Same test tasks before and after training (seed 0, final adapters)

|   i | task               | base                    | A binary                       | C-shuf placebo         | C partial              | B lenient               |
|----:|:-------------------|:------------------------|:-------------------------------|:-----------------------|:-----------------------|:------------------------|
|   0 | [1, 21, 21] -> 42  | 21 + 21 (wrong_numbers) | 21 * (21 / 21) (wrong_numbers) | 21 + 21 * 1 (correct)  | 21 + 21 / 1 (correct)  | 21 + 21 (wrong_numbers) |
|   1 | [20, 23, 16] -> 13 | 23 + 16 - 20 (near)     | 20 + 16 - 23 (correct)         | 23 - 20 + 16 (near)    | 20 + 16 - 23 (correct) | 23 - 20 + 16 (near)     |
|   2 | [12, 13, 5] -> 20  | 13 + 5 (wrong_numbers)  | 12 + 13 - 5 (correct)          | 13 + 12 - 5 (correct)  | 13 + 12 - 5 (correct)  | 13 + 12 - 5 (correct)   |
|   3 | [5, 24, 2] -> 58   | 24 + 5 * 2 (near)       | 24 * (5 - 2) (near)            | 24 + 5 * 2 (near)      | 24 + 5 * 2 (near)      | 24 + 5 * 2 (near)       |
|   4 | [3, 15, 4] -> 16   | 15 + 3 * 4 (near)       | 15 - 3 + 4 (correct)           | 15 + 4 - 3 (correct)   | 15 + 4 - 3 (correct)   | 15 + 4 - 3 (correct)    |
|   5 | [10, 13, 15] -> 18 | 15 + 3 (wrong_numbers)  | 13 + 15 - 10 (correct)         | 15 + 13 - 10 (correct) | 15 + 13 - 10 (correct) | 15 + 13 - 10 (correct)  |
|   6 | [3, 22, 4] -> 23   | 22 + 3 * 4 (near)       | 22 + 4 - 3 (correct)           | 22 + 4 - 3 (correct)   | 22 + 4 - 3 (correct)   | 22 + 4 - 3 (correct)    |
|   7 | [1, 1, 18] -> 17   | 18 + 1 - 1 (near)       | 18 - 1 * 1 (correct)           | 18 - 1 - 1 (near)      | 18 - 1 * 1 (correct)   | 18 - 1 - 1 (near)       |
|   8 | [19, 24, 3] -> 81  | 24 * 3 + 19 (near)      | 24 + 19 * 3 (correct)          | 24 + 19 * 3 (correct)  | 24 + 19 * 3 (correct)  | 24 * 3 + 19 (near)      |
|   9 | [19, 1, 25] -> 44  | 19 + 25 (wrong_numbers) | 25 * 1 + 19 (correct)          | 25 + 19 - 1 (near)     | 25 + 1 * 19 (correct)  | 25 + 19 - 1 (near)      | 

## Error examples (C partial, seed 0, test)

- near: [5, 24, 2] -> 58: `"The arithmetic expression using the numbers [5, 24, 2] and each exactly once, and equals 58 is:\n\n**<answer>24 + 5 * 2</answer>**  \n\nLet's check:  \n24 + 5 * 2 = 24 + 10 = 34 ❌\n\nWait, that's not right. "`
- wrong_numbers: [6, 21, 22] -> 77: `'The arithmetic expression using the numbers [6, 21, 22] that equals 77 is:\n\n**<answer>22 + 21 + 6</answer>**  \n\nThis equals $22 + 21 + 6 = 49$, which is not 77. Let me try again:\n\n**<answer>22 * 21 - '`

## Reward hacking in B: 121 of 2000 test answers get reward 1 but fail the strict check

- [1, 21, 21] -> 42: `21 + 21`
- [1, 23, 13] -> 36: `23 + 13`
- [8, 1, 11] -> 88: `11 * 8`
- [5, 10, 5] -> 75: `10 * 5 + 5 * 5`
- [14, 1, 2] -> 12: `14 - 2`
