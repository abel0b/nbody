[[ $debug = true ]] && set -x

[[ $debug = true ]] && repeat=1 || repeat=10
plot_dir=plot
plot_width=1080
plot_height=720
output_dir=./data/plot
nb_particles=${nb_particles:-10000}
nb_iter=${nb_iter:-10}

set -e

mkdir -p $output_dir
mkdir -p data/bench

echo "performing benchmarks .."

function plot() {
    versions=$@
    name=speedups_${versions// /_}

    echo > $output_dir/$name.dat
    time_ms_ref=none
    for version in $versions
    do
        echo "$version .."
        
        # perf stat -o $version-perf-report.txt -ddd -r $repeat ./bin/nbody-$version
        # time_ms=$(cat $version-perf-report.txt | sed "s/^[ \t]*//" | grep "time elapsed" | cut -d" " -f1 | sed "s/,/\./")
        # TODO: serveral iterations + grep results
        /usr/bin/time -f "%e" ./bin/nbody-$version $nb_particles $nb_iter 2>&1 | tee data/bench/$version.log
        time_ms=$(cat data/bench/$version.log | tail -n1)

        if [[ $time_ms_ref = none ]]
        then
            time_ms_ref=$time_ms
        fi
        speedup=$(echo "print($time_ms_ref/$time_ms)" | python3)
       echo "\"$version\" $speedup" >> $output_dir/$name.dat
    done

    echo > $output_dir/speedups.conf
    echo "set terminal png size $plot_width,$plot_height" >> $output_dir/$name.conf
    echo "set output \"$output_dir/speedups_$name.png\"" >> $output_dir/$name.conf 
    echo "set xlabel \"version\"" >> $output_dir/$name.conf
    echo "set ylabel \"speedup\"" >> $output_dir/$name.conf
    echo "set boxwidth 0.5" >> $output_dir/$name.conf
    echo "set style fill solid" >> $output_dir/$name.conf
    echo "set yrange [0:]" >> $output_dir/$name.conf
    echo "plot \"$output_dir/$name.dat\" using 2: xtic(1) with histogram notitle linecolor rgb '#006EB8'" >> $output_dir/$name.conf
    
    cat $output_dir/$name.conf | gnuplot

    echo "Speedups obtained"
    cat $output_dir/$name.dat
}

plot "baseline baseline-optimized"
plot "baseline baseline-optimized openacc cuda cuda-soa cuda-soa-optimized"
nb_particles=100000
plot "cuda cuda-soa"
plot "cuda cuda-soa cuda-soa-optimized"
