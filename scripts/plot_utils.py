import matplotlib.pyplot as plt
import pickle as pkl

#run_name = 'cxp_densenet_pretrained'
def create_training_curves(run_name):
    run_output_path = '../outputs/{}/'.format(run_name)
    metrics_path = run_output_path + '{}-densenet-{}-metrics.pkl'.format('chex' if 'cxp' in run_name else 'mimic_ch', run_name)
    with open(metrics_path, 'rb') as f:
        metrics = pkl.load(f)

    epochs = []
    trainloss = []
    validauc = []
    for d in metrics:
        epochs.append(d['epoch'])
        trainloss.append(d['trainloss'])
        validauc.append(d['validauc'])

    plt.figure()
    plt.plot(epochs, trainloss)
    plt.ylabel('Train Loss')
    plt.xlabel('Epoch')
    plt.title(run_name + ' Training Curve')
    plt.savefig(run_output_path + '{}-training_curve.png'.format(run_name))

    plt.figure()
    plt.plot(epochs, validauc)
    plt.ylabel('Valid AUC')
    plt.xlabel('Epoch')
    plt.title(run_name + ' Validation Curve')
    plt.savefig(run_output_path + '{}-validation_curve.png'.format(run_name))


if __name__ == '__main__':
    create_training_curves('mimic_densenet_pretrained_v2')