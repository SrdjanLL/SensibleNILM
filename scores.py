import torch
from torch.autograd import Variable
def accuracy_score(net, test_set, batch_size, window_size, std, mean):
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=1)
    accuracy_num = 0
    accuracy_denom = 0
    for i, data in enumerate(testloader, 0):
        inputs, labels = data

        label = labels.mean(dim=1).float()
        #print('Labels: ', label)
        inputs = inputs.view(-1, 1, window_size)
        if torch.cuda.is_available():
            inputs, labels, label = inputs.cuda(), labels.cuda(), label.cuda()

        inputs = Variable(inputs.float())
        outputs = net(inputs)
        outputs = outputs.data * std + mean
        label = label * std + mean
        label = label.view(batch_size, -1)
        distance = torch.abs(outputs - label)
        accuracy_num += sum(distance)
        accuracy_denom += 2 * sum(label)

    score_accuracy = (1 - accuracy_num/accuracy_denom) * 100
    return score_accuracy

def rmse_score(net, test_set, batch_size, window_size, std, mean):
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=1)
    rmse = 0
    for i, data in enumerate(testloader, 0):
        inputs, labels = data

        label = labels.mean(dim=1).float()
        #print('Labels: ', label)
        inputs = inputs.view(-1, 1, window_size)
        if torch.cuda.is_available():
            inputs, labels, label = inputs.cuda(), labels.cuda(), label.cuda()

        inputs = Variable(inputs.float())
        outputs = net(inputs)
        outputs = outputs.data * std + mean
        label = label * std + mean
        label = label.view(batch_size, -1)
        distance = torch.pow(outputs - label, 2)
        rmse += sum(distance)

    rmse = torch.sqrt(rmse/len(test_set))
    return rmse

def mne_score(net, test_set, batch_size, window_size, std, mean):
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=1)
    mne = 0
    label_sum = 0
    output_sum = 0
    for i, data in enumerate(testloader, 0):
        inputs, labels = data

        label = labels.mean(dim=1).float()
        #print('Labels: ', label)
        inputs = inputs.view(-1, 1, window_size)
        if torch.cuda.is_available():
            inputs, labels, label = inputs.cuda(), labels.cuda(), label.cuda()

        inputs = Variable(inputs.float())
        outputs = net(inputs)
        outputs = outputs.data * std + mean
        label = label * std + mean
        label = label.view(batch_size, -1)
        label_sum += sum(label)
        output_sum += sum(outputs)

    mne = torch.abs(output_sum - label_sum)/label_sum
    return mne

def mae_score(net, test_set, batch_size, window_size, std, mean):
        testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=1)
        mae = 0
        for i, data in enumerate(testloader, 0):
            inputs, labels = data

            label = labels.mean(dim=1).float()
            #print('Labels: ', label)
            inputs = inputs.view(-1, 1, window_size)
            if torch.cuda.is_available():
                inputs, labels, label = inputs.cuda(), labels.cuda(), label.cuda()

            inputs = Variable(inputs.float())
            outputs = net(inputs)
            outputs = outputs.data * std + mean
            label = label * std + mean
            label = label.view(batch_size, -1)
            mae += sum(torch.abs(outputs - label))

        mae /= len(test_set)
        return mae

def compare_scores(best_scores, new_scores):
    numerator = 0
    if new_scores['accuracy'] > best_scores['accuracy']:
        numerator += 1
    if new_scores['mae'] < best_scores['mae']:
        numerator += 1
    if new_scores['mne'] < best_scores['mne']:
        numerator += 1
    if new_scores['rmse'] < best_scores['rmse']:
        numerator += 1

    if numerator/len(new_scores) >= 0.75:
        return 1
    else:
        return -1

def get_scores(net, test_set, batch_size, window_size, std, mean):
    net.eval()
    score_accuracy = accuracy_score(net, test_set, 1, window_size, std, mean)
    score_rmse = rmse_score(net, test_set, 1, window_size, std, mean)
    score_mne = mne_score(net, test_set, 1, window_size, std, mean)
    score_mae = mae_score(net, test_set, 1, window_size, std, mean)
    scores= {'accuracy': score_accuracy[0], 'rmse':score_rmse[0], 'mne':score_mne[0],'mae':score_mae[0]}
    print('Accuracy: ', score_accuracy[0], '%')
    print('Root mean squared error: ', score_rmse[0])
    print('Mean Normalized Error: ', score_mne[0])
    print('Mean Absolute Error: ', score_mae[0])
    print('-------------------------------------------\n\n')
    return scores

def test():
	print("You made it")
