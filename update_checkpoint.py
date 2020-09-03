import os
import torch

def update_checkpoint(current_checkpoint, net, clf, critic, epoch, args, script_name, base_optimizer, encoder_optimizer,
                           num_epochs, gradient_penalty , cont_loss , final_loss, test_acc):
    # Save checkpoint.
    print('Saving..')
    current_checkpoint['net'] = net.state_dict()
    current_checkpoint['clf'] = clf.state_dict()
    current_checkpoint['critic'] = critic.state_dict()
    current_checkpoint['epoch'] = epoch
    current_checkpoint['args'] = vars(args)
    current_checkpoint['script'] = script_name
    current_checkpoint['base_optim'] = base_optimizer.state_dict()
    current_checkpoint['encoder_optim'] = encoder_optimizer.state_dict()
    current_checkpoint['num_epochs'] = num_epochs

    # Add gradient penalty, cont_loss, final_loss, test_acc list
    # can check just gradient penalty list
    if 'gradient_penalty' in current_checkpoint:
        current_checkpoint['gradient_penalty'].append(gradient_penalty)
        current_checkpoint['contrastive_loss'].append(cont_loss)
        current_checkpoint['final_loss'].append(final_loss)
        current_checkpoint['test_acc'].append(test_acc)
        current_checkpoint['epoch_list'].append(epoch)
    else:
        current_checkpoint['gradient_penalty'] = [gradient_penalty]
        current_checkpoint['contrastive_loss'] = [cont_loss]
        current_checkpoint['final_loss'] = [final_loss]
        current_checkpoint['test_acc'] = [test_acc]
        current_checkpoint['epoch_list'] = [epoch]


#     print(current_checkpoint)
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    destination = os.path.join('./checkpoint', args.filename + str(epoch))
    torch.save(current_checkpoint, destination)

    return current_checkpoint
