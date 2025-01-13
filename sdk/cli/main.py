from cli.view_logs import view_logs
from cli.plot import plot
from cli.ls import ls
from cli.squeue import squeue

from argparse import ArgumentParser

import logging

logger = logging.getLogger(__name__)


def main():
    setup_logging()

    argumments = parse_arguments()

    if argumments.command == "logs":
        view_logs(
            model_name=argumments.model,
            tail=argumments.tail,
            lines=argumments.lines,
            follow=argumments.follow,
        )

    elif argumments.command == "plot":
        plot(model_name=argumments.model)

    elif argumments.command == "ls":
        ls()

    elif argumments.command == "squeue":
        squeue()

    else:
        logger.error(f"Unknown command {argumments.command}")
        parser.print_help()
        exit(1)


def setup_logging():
    logging.basicConfig(level=logging.DEBUG)

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def parse_arguments():
    parser = ArgumentParser(description="The command line interface for MasInt")

    # Add a subparser for the 'logs' command
    subparsers = parser.add_subparsers(dest="command")

    add_logs_parser(subparsers)
    add_plot_parser(subparsers)
    add_ls_parser(subparsers)
    add_squeue_parser(subparsers)

    args = parser.parse_args()

    return args


def add_logs_parser(subparsers):
    logs_parser = subparsers.add_parser("logs", help="View logs")
    logs_parser.add_argument("--model", help="The model to view logs for")
    logs_parser.add_argument(
        "--tail",
        help="Whether to tail the logs",
        default=False,
        action="store_true",
    )
    logs_parser.add_argument(
        "--follow",
        help="Whether to follow the logs",
        default=False,
        action="store_true",
    )

    logs_parser.add_argument(
        "--lines", type=int, help="The number of lines to view", default=100
    )


def add_plot_parser(subparsers):
    plot_parser = subparsers.add_parser("plot", help="Plot the results of a model")

    plot_parser.add_argument("--model", help="The model to plot results for")


def add_ls_parser(subparsers):
    ls_parser = subparsers.add_parser("ls", help="List models")


def add_squeue_parser(subparsers):
    squeue_parser = subparsers.add_parser("squeue", help="View the squeue")


main()
