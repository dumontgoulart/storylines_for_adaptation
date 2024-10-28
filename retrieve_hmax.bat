echo off

setlocal

rem Set the remote server and username
set server=hmoreno@snellius.surf.nl

REM Set the root directory of the external cluster where the specific subdirectories are located
set "clusterRoot=/gpfs/work3/0/einf4318/paper_4/sfincs/"  REM Change this to your actual cluster path

REM Set the local root directory where you expect the directories to exist
set "localRoot=D:\paper_4\data\sfincs_input"

REM List of subdirectories to process
set "subdirs=idai_ifs_rebuild_bc_hist_rain_surge_noadapt idai_ifs_rebuild_bc_hist_rain_surge_retreat idai_ifs_rebuild_bc_hist_rain_surge_hold idai_ifs_rebuild_bc_3c_rain_surge_noadapt idai_ifs_rebuild_bc_3c_rain_surge_retreat idai_ifs_rebuild_bc_3c_rain_surge_hold idai_ifs_rebuild_bc_hightide_rain_surge_noadapt idai_ifs_rebuild_bc_hightide_rain_surge_retreat idai_ifs_rebuild_bc_hightide_rain_surge_hold idai_ifs_rebuild_bc_3c-hightide_rain_surge_noadapt idai_ifs_rebuild_bc_3c-hightide_rain_surge_retreat idai_ifs_rebuild_bc_3c-hightide_rain_surge_hold"


REM Iterate over the list of subdirectories
for %%s in (%subdirs%) do (
    REM Check if the local directory exists
    if exist "%localRoot%\%%s" (
        REM Use pscp to copy the sfincs_map.nc file from the cluster to the local subdirectory
        echo Copying sfincs_map.nc from %%s to local directory...
        scp %server%:%clusterRoot%%%s/sfincs_map.nc "%localRoot%\%%s"
    ) else (
        echo Local directory %%s does not exist, skipping...
    )
)


echo Done.
endlocal
