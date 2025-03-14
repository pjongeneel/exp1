# test_lambda.py
import json

from main import handler

# Create a mock event similar to what Lambda would receive
mock_event = {
    "body": {
        "input": [1, 2, 3, 4]  # Example input data
    },
    "requestContext": {"identity": {"sourceIp": "127.0.0.1"}},
    # Add other fields that your Lambda function expects
}


# Create a simple mock context object
class MockContext:
    def __init__(self):
        self.function_name = "test-function"
        self.function_version = "$LATEST"
        self.invoked_function_arn = "arn:aws:lambda:us-east-1:123456789012:function:test-function"
        self.memory_limit_in_mb = 128
        self.aws_request_id = "test-request-id"
        self.log_group_name = "/aws/lambda/test-function"
        self.log_stream_name = "2023/01/01/[$LATEST]abcdef123456"
        self.remaining_time_in_millis = 3000

    def get_remaining_time_in_millis(self):
        return self.remaining_time_in_millis


# Run the handler with mock data
mock_context = MockContext()
response = handler(mock_event, mock_context)

# Print the response
print(json.dumps(response, indent=2))
