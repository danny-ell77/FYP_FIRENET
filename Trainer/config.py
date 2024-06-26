import argparse
import json
import os

EVAL_INTERVAL = 300 # every 5 minutes

from . import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
        "--batch_size",
        help ="Batch size for training steps",
        type = int,
        default= 20
    )
    parser.add_argument(
        "--learning_rate",
        help = "Initial learning rate for training",
        type = float,
        default = 0.01
    )
    parser.add_argument(
        "--optimizer",
        help = "optimizing Algorithm for the Loss",
        type = str,
        default= "Adam"
    )
    parser.add_argument(
        "--train_steps",
        help = "Steps to run the training job for. A step is one batch-size",
        type = int,
        default = 100
    )
    parser.add_argument(
        "--output_dir",
        help = "GCS location to write checkpoints and export models",
        required = True
    )
    parser.add_argument(
        "--train_data_path",
        help = "location of train file containing eval URLs",
        default = "gs://cloudfire_lyrical-edition-273206/fire_dataset/tfrecords-dataset-*/"
    )
    parser.add_argument(
        "--eval_data_path",
        help = "location of eval file containing img URLs",
        default = "gs://cloudfire_lyrical-edition-273206/fire_dataset/tfrecords-aug_dataset-9/"
    )
    # build list of model fn"s for help message

    parser.add_argument(
        "--job-dir",
        help="this model ignores this field, but it is required by gcloud",
        default="junk"
    )
    parser.add_argument(
        "--augment",
        help="if specified, augment image data",
        dest="augment",
        action="store_true"
    )
    parser.set_defaults(augment=False)

    # optional hyperparameters used by cnn
    parser.add_argument(
        "--ksize1",
        help = "kernel size of first layer for CNN",
        type = int,
        default = 5
    )
    parser.add_argument(
        "--ksize2",
        help = "kernel size of second layer for CNN",
        type = int,
        default = 5
    )
    parser.add_argument(
        "--ksize3",
        help = "kernel size of third layer for CNN",
        type = int,
        default = 5
    )
    parser.add_argument(
        "--nfil1",
        help = "number of filters in first layer for CNN",
        type = int,
        default = 10
    )

    parser.add_argument(
        "--nfil2",
        help = "number of filters in second layer for CNN",
        type = int,
        default = 20
    )
    parser.add_argument(
        "--nfil3",
        help = "number of filters in third layer for CNN",
        type = int,
        default = 20
    )
    parser.add_argument(
        "--dprob",
        help = "dropout probability for CNN",
        type = float,
        default = 0.25
    )
    parser.add_argument(
        "--batch_norm",
        help="if specified, do batch_norm for CNN",
        dest="batch_norm",
        action="store_false"
    )
    parser.set_defaults(batch_norm=False)

    args = parser.parse_args()
    arg = args.__dict__ # change 'hparams' to 'args' its much intuitive this way 

    output_dir = arg.pop("output_dir")

    # Append trial_id to path for hptuning
    output_dir = os.path.join(
        output_dir,
        json.loads(
            os.environ.get("TF_CONFIG", "{}")
        ).get("task", {}).get("trial", "")
    )

    # Run the training job
    train.train_and_evaluate(output_dir, arg)