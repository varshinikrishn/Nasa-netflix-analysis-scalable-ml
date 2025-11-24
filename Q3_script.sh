#!/bin/bash
#SBATCH --job-name=Q3_script                                        
#SBATCH --time=04:00:00                                                      
#SBATCH --nodes=10                            
#SBATCH --mem=30G                                            
#SBATCH --output=./Output/Q3_output.txt                                       
#SBATCH --mail-user=vkrishnamurthy1@sheffield.ac.uk                                                                              

module load Java/17.0.4
module load Anaconda3/2022.05
source activate myspark

spark-submit /users/acq22vk/com6012/ScalableML/Code/Q3_code.py

