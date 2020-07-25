import os

import torch


def save_checkpoint(net, clf, critic, epoch, args, script_name):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'clf': clf.state_dict(),
        'critic': critic.state_dict(),
        'epoch': epoch,
        'args': vars(args),
        'script': script_name
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    destination = os.path.join('./checkpoint', args.filename)
    torch.save(state, destination)

def save_checkpoint2(net, clf, critic, epoch, args, script_name, base_optimizer, encoder_optimizer, num_epochs):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'clf': clf.state_dict(),
        'critic': critic.state_dict(),
        'epoch': epoch,
        'args': vars(args),
        'script': script_name,
        'base_optim': base_optimizer.state_dict(),
        'encoder_optim': encoder_optimizer.state_dict(),
        'num_epochs': num_epochs
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    destination = os.path.join('./checkpoint', args.filename + str(epoch))
    torch.save(state, destination)
