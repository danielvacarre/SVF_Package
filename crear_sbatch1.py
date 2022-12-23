from os import path, makedirs, listdir
from os.path import isfile, join

if __name__ == '__main__':

    list_method = ["SSVF"]
    list_scenario = ["3"]
    list_size = ["50","70"]

    dir = "./sbatch"
    if not path.exists(dir):
        makedirs(dir)
    for method in list_method:
        for scenario in list_scenario:
            for size in list_size:
                direc = "data_" + scenario + "_" + size
                path_datos = "./data/" + direc
                files = [f for f in listdir(path_datos) if isfile(join(path_datos, f))]
                for f in files:
                    nom_fic = f.split(".")[0]
                    nombre = scenario + "_" + f + "_" + size + "_" + method
                    file = open("./sbatch/" + nom_fic + ".sbatch", "w")
                    print("sbatch " + nom_fic + ".sbatch")
                    file.write("#!/bin/bash\n")
                    file.write("#SBATCH --job-name=" + nom_fic + "\n" +
                               "#SBATCH --ntasks=1" + "\n"+
                               "#SBATCH --qos=normal" + "\n"+
                               "#SBATCH --partition=CLUSTER" + "\n"+
                               "#SBATCH --output=" + nom_fic + ".log" + "\n" +
                               "#SBATCH --error=" + nom_fic + ".err" + "\n" +
                               "#SBATCH --time=24:00:00\n")
                    file.write("source $HOME/python_cplex-3.10/bin/activate\n" +
                               "cd $HOME/Simulaciones_SVF/\n" +
                               "python3  ejecutar_simulacion.py " + direc + " " + f + " " + method + " " + scenario + " 1 " + size + " > " + nom_fic + ".log")
                    file.close()


