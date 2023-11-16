import tritonclient.http


class TritonClient:
    def __init__(self):
        self.url = "localhost:8000"
        self.client = tritonclient.http.InferenceServerClient(self.url)

    def infer(self):
        raise NotImplementedError
