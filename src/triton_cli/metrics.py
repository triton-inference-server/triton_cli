import logging

import requests
from prometheus_client.parser import text_string_to_metric_families

from triton_cli.constants import LOGGER_NAME

# TODO: May make sense to give components unique logger names for debugging
logger = logging.getLogger(LOGGER_NAME)


# No tritonclient metrics API. Only supported through separate metrics HTTP
# endpoint at this time.
class MetricsClient:
    def __init__(self, url="localhost", port=8002):
        if not url:
            url = "localhost"
        if not port:
            port = 8002

        self.url = f"http://{url}:{port}/metrics"

    def get(self, model_name=None):
        r = requests.get(self.url)
        r.raise_for_status()
        metrics_text = r.text
        parsed_families = text_string_to_metric_families(metrics_text)
        metrics = {}
        for family in parsed_families:
            family_metrics = []
            for metric in family.samples:
                md = {
                    "name": metric.name,
                    "labels": metric.labels,
                    "value": metric.value,
                }

                # If model name specified, exclude metrics unrelated to model
                if model_name and metric.labels.get("model") != model_name:
                    continue

                family_metrics.append(md)

            # Exclude empty metrics
            if family_metrics:
                fd = {
                    "name": family.name,
                    "description": family.documentation,
                    "type": family.type,
                    "unit": family.unit,
                    "metrics": family_metrics,
                }
                metrics[family.name] = fd

        return metrics
