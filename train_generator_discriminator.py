import torch
import pandas as pd
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from data_treatment import DataSet, DataAtts
from discriminator import *
from generator import *
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
import glob
from utils import *


class Architecture():
    def __init__(self, learning_rate, batch_size, loss, hidden_layers, name):
        self.learning_rate=learning_rate
        self.batch_size=batch_size
        self.loss=loss
        self.hidden_layers=hidden_layers
        self.name=name

num_epochs=2000
learning_rate=[0.0002]
batch_size=[5]
hidden_layers=[[256]]

architectures=[]
count=0
for lr in learning_rate:
    for b_size in batch_size:
        for hidden in hidden_layers:
            name = str(count)
            name += "_epochs-" + str(num_epochs)
            name += "_layer-" + str(len(hidden))
            name += "_lr-" + str(lr)
            name += "_batch-" + str(b_size)
            name += "_arc-" + ','.join(map(str, hidden))
            architectures.append( Architecture(
                    learning_rate=lr,
                    batch_size=b_size,
                    loss=nn.BCELoss(),
                    hidden_layers=hidden,
                    name=name
                )
            )
            count+=1


file_names=["original_data/diabetes_escalonated.csv",  "original_data/data_escalonated.csv"]
# file_names=["original_data/creditcard_escalonated.csv", "original_data/creditcard.csv"]

# file_names=["original_data/creditcard_escalonated.csv", "original_data/creditcard.csv"]

for file_name in file_names:
    dataAtts = DataAtts(file_name)
    database = DataSet (csv_file=file_name, root_dir=".")
    
    for arc in architectures:
        if ("escalonated" in file_name):
            esc = torch.nn.Sigmoid()
        else:
            esc = False
        generatorAtts = {
            'out_features':dataAtts.class_len, 
            'leakyRelu':0.2, 
            'hidden_layers':arc.hidden_layers,
            'in_features':100, 
            'escalonate':esc
        }
        generator = GeneratorNet(**generatorAtts)

        discriminatorAtts = {
            'in_features':dataAtts.class_len,
            'leakyRelu':0.2,
            'dropout':0.3,
            'hidden_layers':arc.hidden_layers[::-1]
        
        }
        discriminator = DiscriminatorNet(**discriminatorAtts)

        if torch.cuda.is_available():
            discriminator.cuda()
            generator.cuda()
        d_optimizer = optim.Adam(discriminator.parameters(), lr=arc.learning_rate)
        g_optimizer = optim.Adam(generator.parameters(), lr=arc.learning_rate)
        loss = arc.loss
        data_loader = torch.utils.data.DataLoader(database, batch_size=arc.batch_size, shuffle=True)
        num_batches = len(data_loader)
        d_error_plt = [0]
        g_error_plt = [0]

        generated_points = []

        print(arc.name)
        for epoch in range(num_epochs):
            print("Epoch ", epoch)

            for n_batch, real_batch in enumerate(data_loader):
                # 1. Train Discriminator
                real_data = Variable(real_batch).float()
                if torch.cuda.is_available(): 
                    real_data = real_data.cuda()
                # Generate fake data
                fake_data = generator(random_noise(real_data.size(0))).detach()
                # Train D
                d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer, discriminator, loss, real_data, fake_data)

                # 2. Train Generator
                # Generate fake data
                fake_data = generator(random_noise(real_batch.size(0)))
                generated_points.append(fake_data)
                # Train G
                g_error = train_generator(g_optimizer, discriminator, loss, fake_data)

                # Display Progress

                #if (n_batch) % print_interval == 0:
            filename = "results/" + dataAtts.fname + "/" + arc.name + ".txt"
            file = open(filename, "w")

            file.write("Discriminator error: " + str(d_error) + "\n")
            file.write("Generator error: " + str(g_error) + "\n")
            file.write("Points: " + str(fake_data) + "\n\n\n")

            d_error_plt.append(d_error)
            g_error_plt.append(g_error)

        torch.save({
            'epoch': epoch,
            'model_attributes': generatorAtts,
            'model_state_dict': generator.state_dict(),
            'optimizer_state_dict': g_optimizer.state_dict(),
            'loss': loss
            }, "models/" + dataAtts.fname + "/generator_" + arc.name + ".pt")

        torch.save({
            'epoch': epoch,
            'model_attributes': discriminatorAtts,
            'model_state_dict': discriminator.state_dict(),
            'optimizer_state_dict': d_optimizer.state_dict(),
            'loss': loss
            }, "models/" + dataAtts.fname + "/discriminator_" + arc.name + ".pt")


        filename = "results/" + dataAtts.fname + "/error_growth_" + arc.name + ".txt"
        file = open(filename, "w")
        file.write("Discriminator error: " + str(d_error_plt) + "\n")
        file.write("\n\n\n")
        file.write("Generator error: " + str(g_error_plt) + "\n")
        file.close()

        plt.plot(d_error_plt, 'b')
        plt.plot(g_error_plt, 'r')
        plt.savefig('images/'+ dataAtts.fname + "/" + arc.name + '_error.png')
        plt.clf()