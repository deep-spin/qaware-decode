submitter slurm_emulator :: COMMANDS TASK_VARIABLES CONFIGURATION {
    action run > exit_code {
        wrapper="ducttape_job.sh"

        echo "#!/bin/bash" > $wrapper
        echo "set -e # stop on errors" >> $wrapper
        echo "set -o pipefail # stop on pipeline errors" >> $wrapper
        echo "set -u # stop on undeclared variables" >> $wrapper
    
        # print actual jobs 
        echo "$COMMANDS" >> $wrapper
        if (var_defined gres); then
            if [ $gres = "gpu:1" ]; then
                gpu=$(get_free_gpu --is-empty)
            else
                echo "other resource specifications are not implemented"
                exit 1
            fi
        fi
        if (var_defined mem); then
            # TODO
            :
        fi
        if (var_defined time); then
            # TODO
            :
        fi
        if (var_defined nodelist); then
            # TODO
            :
        fi
        if (var_defined exclude); then
            # TODO
            :
        fi

        CUDA_VISIBLE_DEVICES=$gpu bash $wrapper
    }
}