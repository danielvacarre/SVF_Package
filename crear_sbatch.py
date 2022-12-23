from os import path, makedirs

if __name__ == '__main__':

    list_method = ["SSVF","SVF-SP"]
    list_scenario = ["1","2","3"]
    list_size = ["30","50","70","100"]

    dir = "./sbatch"
    if not path.exists(dir):
        makedirs(dir)
    for method in list_method:
        for scenario in list_scenario:
            for size in list_size:
                nombre = scenario + "_" + size + "_" + method
                file = open("./sbatch/" + nombre + ".sbatch", "w")
                print("sbatch " + nombre + ".sbatch")
                file.write("#!/bin/bash\n")
                file.write("#SBATCH --job-name=" + nombre + "\n" +
                           "#SBATCH --ntasks=1" + "\n"+
                           "#SBATCH --qos=xlong" + "\n"+
                           "#SBATCH --partition=CLUSTER" + "\n"+
                           "#SBATCH --output=" + nombre + ".log" + "\n" +
                           "#SBATCH --error=" + nombre + ".err" + "\n" +
                           "#SBATCH --time=240:00:00\n")
                direc = "data_" + scenario + "_" + size
                file.write("source $HOME/python_cplex-3.10/bin/activate\n" +
                           "cd $HOME/Simulaciones_SVF/\n" +
                           "python3  ejecutar_simulaciones.py " + direc + " " + method + " " + scenario + " 1 " + size + " > " + nombre + ".log")
                file.close()


