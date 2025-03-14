def load_model(**kwargs):
    # TODO: load model
    pass


def run_inference(event, model, **kwargs):
    # TODO: run inference
    # input_array = event["body"]["input"] # TODO: get whatever field in your request is the input vector
    # model.run_inference(input_array)
    pass


def handler(event, context):
    print(event)
    print(context)
    # TODO: load model
    # model = load_model()

    # TODO: Process input from event (run inference)
    # result = run_inference(event, model)

    # TODO: Return output (process output if needed)
    # return {"statusCode": 200, "output_classification": "Success"}
    # return {"statusCode": 404, "output_classification": None, "status_reason": "Invalid input (security check failed)"}
    return {"statusCode": 200}
