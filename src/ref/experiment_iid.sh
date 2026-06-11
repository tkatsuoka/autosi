# !/bin/bash
cd $PBS_O_WORKDIR

machin_min=1
machin_max=9

machin_index=0
divisor=$((machin_max))


for category in "iid"; do
    for size in 8 16 32 64; do
        for seed in 0 1 2 3 4 5 6 7 8 9; do
            while true; do
                machin=$(printf "%02d" $((($machin_index % $divisor) + $machin_min)))
                if [ "$machin" != "04" ] && [ "$machin" != "05" ] && [ "$machin" != "06" ]; then
                    break
                fi
                machin_index=$((machin_index + 1))
            done

            if [ $size -eq 8 ]; then
                iter=300
            else
                iter=120
            fi
            qsub -v MYARGS="--category $category --size $size --signal 0.0 --workers 96 --iter $iter --seed $seed --distance 0" "f${machin}.sh"
            machin_index=$((machin_index + 1))
        done
    done
done
