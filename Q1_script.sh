#!/bin/bash
#SBATCH --job-name=Q1_script                                        
#SBATCH --time=02:00:00                                                      
#SBATCH --nodes=2                             
#SBATCH --mem=5G                                            
#SBATCH --output=./Output/Q1_output.txt                                       
#SBATCH --mail-user=vkrishnamurthy1@sheffield.ac.uk                                                                              

module load Java/17.0.4
module load Anaconda3/2022.05
source activate myspark

spark-submit /users/acq22vk/com6012/ScalableML/Code/Q1_code.py



















