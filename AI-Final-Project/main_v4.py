import os
import torch
import torch.nn as nn
from functools import partial
from ray import tune
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import Dataset, random_split

from nn_models.CNN import CNN
from MyLibriSpeech import MyLibriSpeech
from MyCommonVoice import MyCommonVoice


def load_data(dataset_name):
    print(f"Loading {dataset_name} dataset...")
    train_set, val_set, test_set = Dataset(), Dataset(), Dataset()
    if dataset_name == "librispeech":
        # dataset_path = 'data/librispeech/'
        dataset_path = os.path.dirname(os.path.abspath(__file__)) + "/data/librispeech/"
        train_set = MyLibriSpeech(dataset_path, url="train-clean-100", download=True)

        # Calculate the size of the validation set (2% of the train_set)
        validation_size = int(0.02 * len(train_set))

        print("[load_data] dataset size", len(train_set))
        print("[load_data] val size", int(0.02 * len(train_set)))

        train_size = len(train_set) - validation_size
        # Split the train_set into train and validation sets
        train_set, val_set = random_split(train_set, [train_size, validation_size])

        test_set = MyLibriSpeech(dataset_path, url="test-clean", download=True)
        max_length = 392400
        train_set.char2index = {'D': 0, 'G': 1, 'O': 2, 'S': 3, 'J': 4, 'T': 5, ' ': 6, 'E': 7, 'X': 8, 'A': 9, 'K': 10,
                              'R': 11, 'N': 12, 'L': 13, 'U': 14, 'C': 15, 'I': 16, 'M': 17, "'": 18, 'H': 19, 'Q': 20,
                              'B': 21, 'Y': 22, 'Z': 23, 'W': 24, 'F': 25, 'P': 26, 'V': 27}
        test_set.char2index = train_set.char2index
    elif dataset_name == "commonvoice":
        dataset_path = 'data/cv-corpus-13.0-delta-2023-03-09/'
        train_set = MyCommonVoice(dataset_path, train=True)
        train_set.char2index = {'b': 0, 'o': 1, 'c': 2, '.': 3, 'd': 4, 'I': 5, 'X': 6, 'U': 7, '’': 8, 'z': 9, ':': 10,
                                ' ': 11, ',': 12, '?': 13, 'N': 14, 't': 15, 'x': 16, '—': 17, '‑': 18, 'Q': 19,
                                'm': 20, 'A': 21, 'J': 22, 'C': 23, 'P': 24, 'g': 25, '"': 26, ';': 27, 'r': 28,
                                'L': 29, 'h': 30, 'i': 31, 'y': 32, 'k': 33, 'K': 34, 'â': 35, 'e': 36, '”': 37,
                                'v': 38, 'M': 39, 'W': 40, '-': 41, 'T': 42, 's': 43, '(': 44, 'a': 45, 'Y': 46,
                                'l': 47, 'D': 48, '“': 49, 'é': 50, 'G': 51, 'B': 52, "'": 53, 'S': 54, 'q': 55,
                                'w': 56, 'E': 57, '–': 58, 'R': 59, '!': 60, 'j': 61, 'O': 62, 'p': 63, 'Z': 64,
                                'V': 65, 'n': 66, ')': 67, 'F': 68, 'f': 69, '‘': 70, 'u': 71, 'H': 72}

        val_set = MyCommonVoice(dataset_path, train=False)
        val_set.val_set = train_set.val_set
        val_set.process_common_voice_val_set()
        val_set.char2index = train_set.char2index

        test_set = MyCommonVoice(dataset_path, train=False)
        test_set.test_set = train_set.test_set
        test_set.process_common_voice_test_set()
        test_set.char2index = train_set.char2index

        max_length = 352512

    return train_set, val_set, test_set



def load_state_librispeech(config):
    print("[load_state_librispeech]")
    # checkpoint_path = "data/librispeech_pretrained.pth"
    checkpoint_path = os.path.dirname(os.path.abspath(__file__)) + "/data/librispeech_pretrained.pth"
    print("[load_state_librispeech] vai abrir ", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model_state_dict = checkpoint["state_dict"]
    model = CNN(config)
    model.load_state_dict(model_state_dict, strict=False)
    return model


def load_state_commonvoice(config):
    print("[load_state_commonvoice]")
    from speechbrain.pretrained import EncoderDecoderASR
    model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-wav2vec2-commonvoice-en",
                                           savedir="pretrained_models/asr-wav2vec2-commonvoice-en")
    return model


def load_model(config):
    print("[load_model]")
    print(f"Loading {config['dataset']} dataset...")
    if config["dataset"] == "librispeech":
        model = load_state_librispeech(config)
    elif config["dataset"] == "commonvoice":
        model = load_state_commonvoice(config)
    return model


def test_accuracy(net, device="cpu"):
    print("[test_accuracy]")
    train_set, val_set, test_set = load_data(config["dataset"])

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False, num_workers=2)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


# data_dir arg is necessary for trial to work
def train_model(config, data_dir=None):
    print("[train_model]")
    model = load_model(config)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    checkpoint = session.get_checkpoint()

    if checkpoint:
        checkpoint_state = checkpoint.to_dict()
        start_epoch = checkpoint_state["epoch"]
        model.load_state_dict(checkpoint_state["net_state_dict"])
        optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    train_set, val_set, test_set = load_data(config["dataset"])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=int(config["batch_size"]), shuffle=True,
                                               num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=int(config["batch_size"]), shuffle=True, num_workers=8)

    for epoch in range(start_epoch, 10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            print("[train_model] data", data)
            inputs, labels = data
            print("[train_model] inputs", inputs.shape)
            # inputs = torch.tensor(inputs)
            # labels = torch.tensor(labels)
            # inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(val_loader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        checkpoint_data = {
            "epoch": epoch,
            "net_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        checkpoint = Checkpoint.from_dict(checkpoint_data)

        session.report({"loss": val_loss / val_steps, "accuracy": correct / total}, checkpoint=checkpoint)
    print("Finished Training")


def main(config):
    print("[main]")

    assert config["dataset"] in ("librispeech", "commonvoice")

    data_dir = os.path.abspath("./data")

    config["l1"] = tune.choice([2**i for i in range(config["num_samples"] - 1)])
    config["l2"] = tune.choice([2**i for i in range(config["num_samples"] - 1)])
    config["lr"] = tune.loguniform(1e-4, 1e-1)
    config["batch_size"] = tune.choice([2, 4, 8, 16, 32])

    scheduler = ASHAScheduler(metric="loss", mode="min", max_t=config["epochs"], grace_period=1, reduction_factor=2)
    result = tune.run(partial(train_model, data_dir=data_dir),
                      resources_per_trial={"cpu": 2, "gpu": config["gpus_per_trial"]}, config=config,
                      num_samples=config["num_samples"], scheduler=scheduler)

    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")

    config["l1"] = best_trial.config["l1"]
    config["l2"] = best_trial.config["l2"]
    best_trained_model = load_model(config)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if config["gpus_per_trial"] > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint = best_trial.checkpoint.to_air_checkpoint()
    best_checkpoint_data = best_checkpoint.to_dict()

    best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])

    test_acc = test_accuracy(best_trained_model, device)
    print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == "__main__":
    config = {
        "epochs": 10,
        "gpus_per_trial": 0,
        "num_samples": 2
    }

    # for dataset_name in ["librispeech", "commonvoice"]:
    #     config["dataset"] = dataset_name
    #     main(config=config)

    config["dataset"] = "librispeech"
    main(config=config)
