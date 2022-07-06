# ntth-comparisons
--------------------------------------------------

------------
Introduction
------------
Comparisons of algorithms and models implemented on different hardware/software platforms (FPAA, SpiNNaker 2, CPU, GPU, etc.).

-----------------------
Repository Organization
-----------------------

    +-- \ntth-comparisons       Top level 
         |
         +-- \tasks             Benchmark tasks
         |    |              
         |    +--\<task_name>   Name this folder with the task title.
         |        |                             
         |        +--\<framework_1>     Name this folder with the
         |        |    |                framework of the implementation.
         |        |    |    
         |        |    +--\<target_HW_1>    Name this folder with the 
         |        |    |                    target hardware.
         |        |    | 
         |        |    +--\<target_HW_2>    Another hardware...
         |        |    |
         |        |    +-- ...
         |        |
         |        +--\<framework_2>     Another framework...
         |        |    |
         |        |    +-- ...
         |        |
         |        +-- ...     
         |
         +-- LICENSE            License file.
         |
         +-- README.md          This file.

----------------------
Copyright and Licenses
----------------------

See the license in the LICENSE file.