import os

import torch


def save_checkpoint(net, clf, critic, epoch, args, script_name, results):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'clf': clf.state_dict(),
        'critic': critic.state_dict(),
        'epoch': epoch,
        'args': vars(args),
        'script': script_name,
        'results': results
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    destination = os.path.join('./checkpoint', args.filename + '_epoch{:03d}.pth'.format(epoch))
    torch.save(state, destination)