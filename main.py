#import required libs
import torch
#from process_trail import process_trail_log
import numpy as np
import sys
from argparse import Namespace
from pytorch_lightning import LightningModule
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset

#classifies a single datapoint
def run_inference(datapoint, model, device):

    # Initialize classification to -1 (invalid)
    classification = -1

    #try to classify the datapoint
    try:
        # Ensure the input is a numpy array with the correct dtype
        feature = np.array(datapoint, dtype=np.float32) 
        feature = feature[0:-1]  # Mimic dataset slicing
        feature = np.expand_dims(feature, axis=0)  # Mimic dataset transformation

        # Convert to PyTorch tensor and move to the correct device
        inputs = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dimension

        # Perform inference
        with torch.no_grad():
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # Get class index with highest probability
    
        classification = predicted.item()  # Return predicted class as an integer
    finally:
        return classification

def load_model():
    # Force Python to search in the current directory
    sys.path.append(".")
    
    hparams = Namespace(
        input_dim=91,
        c1_out=64,
        c1_kernel=64,
        c2_out=128,
        c2_kernel=128,
        c3_out=256,
        c3_kernel=256,
        final_conv=512,
        final_kernel=512,
        out=2,
        train_data_path="testCloud/train.npy",
        val_data_path="testCloud/val.npy",
    )

    #Create a new model instance
    model = LuNet(hparams)

    #Load the state dictionary
    model.load_state_dict(torch.load("cmodel_state_dict.pth"))

    # Convert model to evaluation mode
    model.eval()


    # Load the model, overriding the original module path
    #model = torch.load("entire_cmodel.pth", weights_only=False, map_location=torch.device("cpu"), pickle_module=__import__("pickle"))

    print("Model loaded successfully!")
    return model


def handler(event, context):
    print(event)
    print(context)

    # Force PyTorch to use CPU only
    torch.set_default_tensor_type(torch.FloatTensor)

    print("Running on:", torch.device("cpu"))

    
    # load model
    model = load_model()

    # load model and push to device (use cpu as lambda only supports cpu)
    device = torch.device("cpu")
    model.to(device)
    print("model primed")
    
    # Process input from event
    processed_data = process_trail_log(event)
    
    #send processed input to be classified
    result = run_inference(processed_data, model, device)

    result = 0

    #process output and return result
    if (result < 0):
        return {"statusCode": 404, "output_classification": None, "status_reason": "Invalid input"}
    else:
        ret_string = "Success: with classID: " + str(result)
        return {"statusCode": 200, "output_classification": ret_string}

class LuNetBlock(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=hparams.conv_out,
                kernel_size=hparams.kernel
            ),
            nn.ReLU()
        )
        self.max_pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=2),
            nn.ReLU()
        )
        # calc the output size after max_pool
        dummy_x = torch.randn(1, 1, hparams.input_dim, requires_grad=False)
        dummy_x = self.conv(dummy_x)
        dummy_x = self.max_pool(dummy_x)
        max_pool_out = dummy_x.shape[1]
        lstm_in = dummy_x.shape[2]

        self.batch_norm = nn.BatchNorm1d(max_pool_out)

        self.lstm = nn.LSTM(lstm_in, hparams.conv_out)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.batch_norm(x)

        # lstm and relu
        x, hidden = self.lstm(x)
        x = F.relu(x)

        # reshape
        x = x.view(x.shape[0], 1, -1)

        # drop out
        x = self.dropout(x)

        return x


class LuNet(LightningModule):
    def __init__(self, hparams):
        super().__init__()

        # config
        self.train_data_path = hparams.train_data_path
        self.val_data_path = hparams.val_data_path
        self.out_dim = hparams.out

        hparams_lu_block_1 = Namespace(**{
            'input_dim': hparams.input_dim,
            'conv_out': hparams.c1_out,
            'kernel': hparams.c1_kernel
        })
        self.lu_block_1 = LuNetBlock(hparams_lu_block_1)

        # use dummy to calc output
        dummy_x = torch.randn(1, 1, hparams.input_dim, requires_grad=False)
        dummy_x = self.lu_block_1(dummy_x)

        hparams_lu_block_2 = Namespace(**{
            'input_dim': dummy_x.shape[2],
            'conv_out': hparams.c2_out,
            'kernel': hparams.c2_kernel
        })
        self.lu_block_2 = LuNetBlock(hparams_lu_block_2)

        dummy_x = self.lu_block_2(dummy_x)

        hparams_lu_block_3 = Namespace(**{
            'input_dim': dummy_x.shape[2],
            'conv_out': hparams.c3_out,
            'kernel': hparams.c3_kernel
        })
        self.lu_block_3 = LuNetBlock(hparams_lu_block_3)

        dummy_x = self.lu_block_3(dummy_x)

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=hparams.final_conv,
                kernel_size=hparams.final_kernel,
            )
        )

        dummy_x = self.conv(dummy_x)
        self.avg_pool = nn.AvgPool1d(kernel_size=hparams.final_conv)

        dummy_x = self.avg_pool(dummy_x)
        self.drop_out = nn.Dropout(p=0.5)

        if self.out_dim == 2:  # binary classification
            self.out = nn.Sequential(
                nn.Linear(
                    in_features=dummy_x.shape[1] * dummy_x.shape[2],
                    out_features=1

                ),
                nn.Sigmoid()
            )
        else:
            self.out = nn.Linear(
                in_features=dummy_x.shape[1] * dummy_x.shape[2],
                out_features=self.out_dim
            )

    def forward(self, x):
        x = self.lu_block_1(x)
        x = self.lu_block_2(x)
        x = self.lu_block_3(x)
        x = self.conv(x)
        x = self.avg_pool(x)
        x = self.drop_out(x)

        # reshape
        x = x.view(x.shape[0], -1)

        x = self.out(x)

        return x

    def train_dataloader(self):
        data_loader = DataLoader(CloudDataset(self.train_data_path), batch_size=16, shuffle=True, num_workers=4)

        return data_loader

    def val_dataloader(self):
        data_loader = DataLoader(CloudDataset(self.val_data_path), batch_size=16, shuffle=True, num_workers=4)

        return data_loader

    def training_step(self, batch, batch_idx):
        x = batch['feature'].float()
        y_hat = self(x)

        if self.out_dim == 2:  # binary classification
            y = batch['label'].float()
            y = y.unsqueeze(1)
            loss = {'loss': F.binary_cross_entropy(y_hat, y)}
        else:
            y = batch['attack_cat'].long()
            loss = {'loss': F.cross_entropy(y_hat, y)}

        if (batch_idx % 50) == 0:
            self.logger.log_metrics(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['feature'].float()
        y_hat = self(x)

        if self.out_dim == 2:  # binary classification
            y = batch['label'].float()
            y = y.unsqueeze(1)
            loss = {'val_loss': F.binary_cross_entropy(y_hat, y)}
        else:
            y = batch['attack_cat'].long()
            loss = {'val_loss': F.cross_entropy(y_hat, y)}

        if (batch_idx % 50) == 0:
            self.logger.log_metrics(loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.parameters(), lr=0.001)

class CloudDataset(Dataset):
    def __init__(self, data_path):
        self.data = np.load(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
      return {'feature':  np.expand_dims(self.data[idx, 0:-2], axis=0), 'label': self.data[idx, -1]}

possible_user_types = ['IAMUser', 'AssumedRole', 'AWSAccount', 'Root', 'AWSService']
possible_user_arn_type = ["assumed-role/config", "assumed-role/level6", "assumed-role/lambda", "assumed-role/", "root", "level5", "Cloudsploit", "user/backup", "user/"]
possible_user_agent_type = ["aws-cli/", "Boto3/", "aws-sdk", "AWS Console Lambda", "aws-iternal/", "awslambda", "Boto/", "BotoCore"]
possible_event_type = ["AwsConsoleAction", "AwsConsoleSignIn", "AwsApiCall"]
possible_event_names = ['GetSdk', 'GetApi', 'PutMethodResponse', 'GetLoginProfile', 'GetBucketReplication', 'GetFindings', 'GetKeyPairs', 'GetRoutes',
                        'GetRandomPassword', 'GetPasswordData', 'GetKeyRotationStatus', 'GetIntegrationResponse',
                        'GetBucketNotification', 'GetTemplate', 'GetConnectors', 'GetRestApis', 'BatchGetRepositories', 'GetGatewayResponse',
                        'GetExportSnapshotRecords', 'GetMethod', 'GetDocumentationParts', 'GetSessionToken', 'GetOperations', 'GetDomainNames', 'GetDisks',
                       'GetBucketWebsite', 'PutMethod', 'GetConnections', 'PutBucketTagging', 'GetRelationalDatabases', 'GetBlueprints']
possible_region = ['ap-south-1', 'us-east-1', 'eu-west-1', 'eu-west-2', 'us-west-1',
                   'sa-east-1', 'eu-central-1', 'us-east-2', 'ca-central-1', 'eu-west-3',
                   'ap-northeast-3', 'ap-southeast-1', 'ap-southeast-2', 'eu-north-1',
                   'us-west-2', 'ap-northeast-1', 'ap-northeast-2']
possible_event_source = ['support.amazonaws.com', 'lambda.amazonaws.com', 'iam.amazonaws.com', 'sagemaker.amazonaws.com', 'elasticache.amazonaws.com', 'rds.amazonaws.com', 'monitoring.amazonaws.com',
                         'elasticbeanstalk.amazonaws.com', 'kinesis.amazonaws.com', 'cloudfront.amazonaws.com', 's3.amazonaws.com', 'signin.amazonaws.com', 'logs.amazonaws.com', 'dynamodb.amazonaws.com',
                         'apigateway.amazonaws.com', 'events.amazonaws.com', 'cloudtrail.amazonaws.com']


#helper function to vectorize
def oneHotVectorize(possibleParams, current_value, stringMatch):
  # Create a zero vector of length equal to number of possible values
    one_hot_vector = np.zeros(len(possibleParams), dtype=int)

    # Ensure current_value is a string before substring matching
    if stringMatch:
        current_value = str(current_value)  # Convert to string to prevent errors

        for i, s in enumerate(possibleParams):
            if isinstance(s, float):  # Handle possible float values
                continue  # Skip float values since they cannot be substrings

            s = str(s)  # Convert s to string to allow comparison
            if s in current_value:
                one_hot_vector[i] = 1
                break  # Stop at first match

    else:
        if current_value in possibleParams:
            index = possibleParams.index(current_value)
            one_hot_vector[index] = 1

    return one_hot_vector


def process_trail_log(trail_request):
    #sample event:
    #result = np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,10.15,0], dtype=np.float32)
    #print("size of test data: ", str(len(result)))
    result = np.array(trail_request['testdatapoint'], dtype=np.float32)
    return result
