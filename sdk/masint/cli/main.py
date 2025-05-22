from masint.cli.view_logs import view_logs
from masint.cli.plot import plot
from masint.cli.ls import ls
from masint.cli.squeue import squeue
from masint.cli.stats import stats

import masint

from argparse import ArgumentParser

import logging

logger = logging.getLogger(__name__)


def main():
    setup_logging()

    argumments, parser = parse_arguments()

    if argumments.command == "logs":
        view_logs(
            model_name=argumments.model,
            tail=argumments.tail,
            lines=argumments.lines,
            follow=argumments.follow,
        )

    elif argumments.command == "plot":
        plot(model_name=argumments.model, smooth=int(argumments.smooth))

    elif argumments.command == "ls":
        ls(all=argumments.all, limit=argumments.limit)

    elif argumments.command == "squeue":
        squeue()

    elif argumments.command == "stats":
        stats()

    else:
        logger.error(f"Unknown command {argumments.command}")
        parser.print_help()
        exit(1)


def setup_logging():
    logging.basicConfig(level=logging.WARNING)

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def parse_arguments():
    parser = ArgumentParser(description="The command line interface for MasInt ScalarLM")

    # Add a subparser for the 'logs' command
    subparsers = parser.add_subparsers(dest="command")

    add_logs_parser(subparsers)
    add_plot_parser(subparsers)
    add_ls_parser(subparsers)
    add_squeue_parser(subparsers)
    add_stats_parser(subparsers)

    args = parser.parse_args()

    return args, parser


def add_logs_parser(subparsers):
    logs_parser = subparsers.add_parser("logs", help="View logs")
    logs_parser.add_argument("--model", help="The model to view logs for", default="latest")
    logs_parser.add_argument(
        "--tail",
        help="Whether to tail the logs",
        default=False,
        action="store_true",
    )
    logs_parser.add_argument(
        "--follow",
        "-f",
        help="Whether to follow the logs",
        default=False,
        action="store_true",
    )

    logs_parser.add_argument(
        "--lines", type=int, help="The number of lines to view", default=100
    )


def add_plot_parser(subparsers):
    plot_parser = subparsers.add_parser("plot", help="Plot the results of a model")

    plot_parser.add_argument("--model", help="The model to plot results for", default="latest")
    plot_parser.add_argument("--smooth", help="The number of steps to smooth over", default=1)


def add_ls_parser(subparsers):
    ls_parser = subparsers.add_parser("ls", help="List models")
    ls_parser.add_argument("-A", "--all", help="List all attributes of the models", default=False, action="store_true")
    ls_parser.add_argument("-l", "--limit", help="Limit the number of models returned", default=None, type=int)


def add_squeue_parser(subparsers):
    squeue_parser = subparsers.add_parser("squeue", help="View the squeue")

def add_stats_parser(subparsers):
    stats_parser = subparsers.add_parser("stats", help="View the stats of the models")


if __name__ == "__main__":
    main()
