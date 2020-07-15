import torch
from vgg import *
from torch.utils.data import DataLoader, TensorDataset
from catalyst import dl
import numpy as np
print('data')
# data
num_samples, num_features = int(1e4), int(1e1)
print('num',num_samples,num_features)
X, y = torch.rand(num_samples, num_features), torch.rand(num_samples)
print(X.shape, y.shape)
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, num_workers=1)
loaders = {"train": loader, "valid": loader}

print('model')

# model, criterion, optimizer, scheduler
model = vgg13().cuda()
criterion = nn.BCELoss().cuda()
optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [3, 6])

print('training')

# model training
runner = dl.SupervisedRunner()
logdir = './logdir'
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    loaders=loaders,
    logdir=logdir,
    num_epochs=1,
    verbose=True,
    callbacks=[dl.BatchOverfitCallback(train=10, valid=0.5)]
)

runner.valid_metrics = {"loss": criterion}
runnerpredictions = np.vstack(list(map(
    lambda x: x["logits"].cpu().numpy(), 
    runner.predict_loader(loader=loaders["valid"], resume=f"{logdir}/checkpoints/best.pth")
)))

print(runnerpredictions.shape)

secondRunnerPredictions = list(
    map(
        lambda x:x, 
        loaders["valid"]
        )
    )

for el in secondRunnerPredictions:
    print('el0', el[0].shape)
    print('el1', el[1].shape)
    print('len el', len(el))
    print('model',model(el[0]).shape)
    break


print(len(secondRunnerPredictions))
