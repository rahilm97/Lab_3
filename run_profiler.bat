@echo off
SET pathToOriginal=".\Cuda Original Addition\x64\Debug\Lab_2.exe"
SET pathToAdditionGlobal=".\Cuda Addition Global\x64\Debug\Cuda Addition Global.exe"
SET pathToVectorAddReg=".\Vector Addition Registers\x64\Debug\Vector Addition Registers.exe"
SET pathToVectorAddShared=".\Vector Addition Shared\x64\Debug\Vector Addition Shared.exe"

IF [%1]==[] IF [%2]==[] IF [%3]==[] IF [%4]==[] GOTO usage

IF %1==0 GOTO run_original
IF %1==1 GOTO run_addition_global
IF %1==2 GOTO run_vector_add_register
IF %1==3 GOTO run_vector_add_shared

:run_original
    ECHO Attempting to perform profiling on 'Cuda Original Addition'
    IF EXIST %pathToOriginal% (
        ECHO Starting Profiler on 'Cuda Original Addition' executable...
        nv-nsight-cu-cli.bat -f -o original -k cuda_calculator %pathToOriginal% %2 %3 %4 q
        ECHO Profiler complete...
    ) else (
        ECHO %pathToOriginal% does not exist or cannot be accessed.
    )
    GOTO :EOF

:run_addition_global
    ECHO Attempting to perform profiling on 'Cuda Addition Global'
    IF EXIST %pathToAdditionGlobal% (
        ECHO Starting Profiler on 'Cuda Addition Global' executable...
        nv-nsight-cu-cli -f -o addition_global -k cuda_calculator %pathToAdditionGlobal% %2 %3 %4 q
        ECHO Profiler complete...
    ) ELSE (
        ECHO skipping
        ECHO %pathToAdditionGlobal% not found. Skipping profiler run
    )
    GOTO :EOF

:run_vector_add_register
    ECHO Attempting to perform profiling on 'Vector Addition Registers'
    IF EXIST %pathToVectorAddReg% (
        ECHO Starting Profiler on 'Vector Addition Registers' executable...
        nv-nsight-cu-cli -f -o vector_add_register -k cuda_calculator %pathToVectorAddReg% %2 %3 %4 q
        ECHO Profiler complete...
    ) ELSE (
        ECHO %pathToVectorAddReg% not found. Skipping profiler run
    )
    GOTO :EOF

:run_vector_add_shared
    ECHO Attempting to perform profiling on 'Vector Addition Shared'
    IF EXIST %pathToVectorAddShared% (
        ECHO Starting Profiler on 'Vector Addition Shared' executable...
        nv-nsight-cu-cli -f -o vector_add_shared -k cuda_calculator %pathToVectorAddShared% %2 %3 %4 q
        ECHO Profiler complete...
    ) ELSE (
        ECHO %pathToVectorAddShared% not found. Skipping profiler run
    )
    GOTO :EOF

:usage
    ECHO Usage for run_profiler.bat
    ECHO run_profiler.bat `MODE` `Size of N` `Number of Blocks` `Number of Threads`
    ECHO    MODE = {
    ECHO        0 - Profiles the 'Cuda Original Addition' x64 debug executable.
    ECHO            Generates a original.nsight-cuprof-report report file.
    ECHO        1 - Profiles the 'Cuda Addition Global' x64 debug executable
    ECHO            Generates a addition_global.nsight-cuprof-report report file.
    ECHO        2 - Profiles the 'Vector Addition Registers' x64 debug executable
    ECHO            Generates a vector_add_register.nsight-cuprof-report report file.
    ECHO        3 - Profiles the 'Vector Addition Shared' x64 debug executable
    ECHO            Generates a vector_add_shared.nsight-cuprof-report report file.
    ECHO    }
    ECHO WARNING!!: Everytime this script is ran, it will forcefully overwrite the old report file with the new one you are trying to create
    GOTO :EOF