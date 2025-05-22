
import time

generate_metrics = None

class Metrics:
    def __init__(self):
        self.queue_depth = 0
        self.epoch_time = None

        self.total_completed_requests = 0
        self.total_completed_tokens = 0
        self.total_completed_flops = 0

        self.total_completed_response_time = 0.0

    def record_completed_request(self, token_count: int, flop_count: int):
        """
        Record a completed request.
        """
        self.total_completed_requests += 1

        if flop_count is not None:
            self.total_completed_flops += flop_count

        if token_count is not None:
            self.total_completed_tokens += token_count

        self.queue_depth -= 1

        current_time = time.time()
        time_since_epoch = current_time - self.epoch_time

        self.total_completed_response_time += time_since_epoch

        self.epoch_time = current_time

    def record_new_request(self):
        """
        Record a new request.
        """
        if self.queue_depth == 0:
            self.epoch_time = time.time()

        self.queue_depth += 1

    def get_all_metrics(self):
        """
        Get the current metrics.
        """
        return {
            "queue_depth": self.queue_depth,
            "total_completed_requests": self.total_completed_requests,
            "total_completed_tokens": self.total_completed_tokens,
            "total_completed_response_time": self.total_completed_response_time,
            "tokens_per_second": self.total_completed_tokens / self.total_completed_response_time if self.total_completed_response_time > 0 else 0,
            "requests_per_second": self.total_completed_requests / self.total_completed_response_time if self.total_completed_response_time > 0 else 0,
            "flops_per_second": self.total_completed_flops / self.total_completed_response_time if self.total_completed_response_time > 0 else 0,
        }


def get_metrics() -> Metrics:
    """
    Get the metrics object.
    """
    global generate_metrics
    if generate_metrics is None:
        generate_metrics = Metrics()
    return generate_metrics
