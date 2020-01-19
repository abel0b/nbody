[[ $debug = true ]] && set -x

versions="baseline openacc cuda cuda-soa"
[[ $debug = true ]] && repeat=1 || repeat=10
plot_dir=plot
plot_width=1080
plot_height=720
output_dir=./data/plot
nb_particles=${nb_particles:-10000}
nb_iter=${nb_iter:-5}

set -e

mkdir -p $output_dir
mkdir -p data/bench

echo "performing benchmarks .."

function plot() {
    echo > $output_dir/speedups.dat
    time_ms_ref=none
    for version in $versions
    do
        echo "$version .."
        
        # perf stat -o $version-perf-report.txt -ddd -r $repeat ./bin/nbody-$version
        # time_ms=$(cat $version-perf-report.txt | sed "s/^[ \t]*//" | grep "time elapsed" | cut -d" " -f1 | sed "s/,/\./")
        /usr/bin/time -f "%e" ./bin/nbody-$version $nb_particles $nb_iter 2>&1 | tee data/bench/$version.log
        time_ms=$(cat data/bench/$version.log | tail -n1)

        if [[ $time_ms_ref = none ]]
        then
            time_ms_ref=$time_ms
        fi
        speedup=$(echo "print($time_ms_ref/$time_ms)" | python3)
       echo "\"$version\" $speedup" >> $output_dir/speedups.dat
    done

    echo > $output_dir/speedups.conf
    echo "set terminal png size $plot_width,$plot_height" >> $output_dir/speedups.conf
    echo "set output \"$output_dir/speedups.png\"" >> $output_dir/speedups.conf 
    echo "set xlabel \"version\"" >> $output_dir/speedups.conf
    echo "set ylabel \"speedup\"" >> $output_dir/speedups.conf
    echo "set boxwidth 0.5" >> $output_dir/speedups.conf
    echo "set style fill solid" >> $output_dir/speedups.conf
    echo "plot \"$output_dir/speedups.dat\" using 2: xtic(1) with histogram notitle" >> $output_dir/speedups.conf
    
    cat $output_dir/speedups.conf | gnuplot

    echo "Speedups"
    cat $output_dir/speedups.dat
}

plot
