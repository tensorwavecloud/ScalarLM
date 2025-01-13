# squeue

```console
./cray llm squeue
```

This command is a wrapper around the `squeue` command. It is used to display the status of jobs in the training queue. The output is similar to the `squeue` command, but with some additional formatting.

```console
             JOBID PARTITION         NAME     USER    STATE       TIME TIME_LIMI  NODES NODELIST(REASON)
                 8     short 00f186ab039b     root  PENDING       0:00     20:00      1 (Priority)
                 7     short f1ba9c0eb11b     root  PENDING       0:00     20:00      1 (Priority)
                 6     short 0746261fd1db     root  PENDING       0:00     20:00      1 (Priority)
                 5     short ae55dedbb496     root  PENDING       0:00     20:00      1 (Priority)
                 4     short d2bc30a36081     root  PENDING       0:00     20:00      1 (Priority)
                 3     short bce8e63a7bef     root  PENDING       0:00     20:00      1 (Resources)
                 2     short c42b59ab0fb1     root  RUNNING       0:34     20:00      1 df294b9206ff
```


