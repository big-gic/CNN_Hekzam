import pandas as pd




class Network :

    def __init__(self, conf) :
        self.conf = conf
        self.params = None


    def load_params(self) :
        self.params = pd.read_json(self.conf["config_path"])


    def display_layers(self) :
        print("Layers :")
        for layer in self.params.at[0, "layer_list"] :
            print(layer)

    def display_transforms(self) :
        print("Data transformations :")
        for trans in self.params.at[0, "data_transformations"]:
            print(trans)

    def display_name(self) :
        print(self.params.at[0, "net"])

    def display(self) :
        self.load_params()
        self.display_name()
        self.display_layers()
        self.display_transforms()
        return

    
    