1. Create a docker repo in ECR 
`aws ecr create-repository --repository-name lambda_inference || true`
2. clone this repo
3. build the docker image
`docker build -t {AWS_ACCOUNT_NUMBER}.dkr.ecr.{AWS_REGION}.amazonaws.com/lambda_inference:latest .`
4. push the docker image
`docker push {AWS_ACCOUNT_NUMBER}.dkr.ecr.{AWS_REGION}.amazonaws.com/lambda_inference:latest`
5. create a lambda function in AWS (one time only)
6. update the lambda function with the new image
`aws lambda update-function-code --function-name lambda_inference --image-uri {AWS_ACCOUNT_NUMBER}.dkr.ecr.{AWS_REGION}.amazonaws.com/lambda_inference:latest`

sleep for 30 seconds to allow update to take effect

`FUNCTION_VERSION=$(aws lambda publish-version --function-name lambda_inference | jq -r '.Version')`

sleep for 30 seconds to allow update to take effect

`aws lambda update-alias --function-name lambda_inference --name latest --function-version $FUNCTION_VERSION`

7. with any code change, rebuild the docker image and push it to ECR (repeat 3 and 4 and 6)



Notes:

If you want to add dependencies 
`uv add scikit-learn`

If you want to test your function before docker
`uv sync`
`python run test.py`
