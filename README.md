The main directories that should be used are Q_Score_PCE (has code to send serial jobs and basic job-arrays) and Q_Score_PCE_Parallel (has a better parallelization implementation). The other documents and directories have previous calculations and can be overlooked.

This code has been designed to run on CESGAs infrastructures, Finisterrae III (HPC computer) anda QMIO (quantum computer), 
some adaptations on the .sh archives should be made to be able to run on different hardware.

There are 3 main Python archives in the Q_Score_PCE directorie:

- qscore.py has the main code that combines the Q-Score benchmark with an implementation of the PCE algorithm
- run_qscore_PCE_single_n.py is what the .sh calls to send jobs via job-array
- aggregate.py gets the data from the .h5 from the calculations and generates a .csv that has the information for multiple 'n',
  using it to create a graph with the information

This aggregate.py functions just with the .h5, being able to combine data from different simulations to create one graph; 
this has been done to be able to save as much information as possible from every run, being able to complete simulations that were interrupted
or in case you want to add extra points to the graph. In case different parameters were used, a notification will appear.

Some other variations that use more parallelization methods are found in the Q_Score_PCE_Parallel directorie, where an extra merge_chunks.py is needed before aggregate.py.
The code is commented with explanations of each function and has indicated how to run it.

When runing the code, 3 directories are created, one called Imágenes, one called logs, and another one called Resultados using the timestamp of the job as name.

In logs, the loggers from slurm are stored, the Resultados directorie stores the results calculated in parallel in .h5 format. 
aggregate.py can combine the loggers in one archive when creating the grph with the results.

The Imágenes directorie serves to store the graphs generated after the use of aggregate.py.
