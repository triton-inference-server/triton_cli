# Copyright 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import json
import logging

import requests
from rich.table import Table, Column
from rich.console import Console
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

    def __parse(self, metrics_text, model_name=None):
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

    def get(self, model_name=None):
        r = requests.get(self.url)
        r.raise_for_status()
        metrics_text = r.text
        metrics = self.__parse(metrics_text, model_name)
        return metrics

    def display_table(self, model_name=None):
        metrics = self.get(model_name=model_name)
        title = "Model Metrics" if model_name else "Server Metrics"
        # Rich formatting
        console = Console()
        table = Table(
            Column("Metric"),
            Column("Description"),
            Column("Labels"),
            Column("Value"),
            show_header=True,
            show_lines=True,
            title=title,
            title_style="bold green",
            highlight=True,
        )

        for family_name in sorted(metrics.keys()):
            family = metrics[family_name]
            for metric in family["metrics"]:
                name = metric["name"]
                # Save width on types with no derivatives like _sum, _total
                if family["type"] in ["counter", "gauge"]:
                    name = family_name

                labels = metric["labels"]
                # Truncate GPU UUIDs to shrink width, first chunk should still
                # be unique and identifiable
                if "gpu_uuid" in labels:
                    truncated_id = "-".join(labels["gpu_uuid"].split("-")[:2])
                    labels["gpu_uuid"] = truncated_id

                # Use json indent formatting to shrink width
                labels = json.dumps(metric["labels"], indent=2)

                table.add_row(name, family["description"], labels, str(metric["value"]))

        if table.rows:
            console.print(table)
        else:
            logger.error("Metrics table was empty")
