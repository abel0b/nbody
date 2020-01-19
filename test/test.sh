[[ debug = "true" ]] && set -x

nb_particles=${nb_particles:-1000}
nb_iter=${nb_iter:-10}

reset="\e[0m"
bold="\e[1mB"
green="\e[32m"
red="\e[31m"

echo "testing all nbody versions"
echo "nb_particles=$nb_particles"
echo "nb_iter=nb_iter"

versions="baseline openacc cuda cuda-soa"

function test() {
    pass=true

    for version in $@
    do
        ./bin/nbody-$version $nb_particles $nb_iter > /dev/null
        if [[ ! "$?" = "0" ]]
        then
            echo "./bin/nbody-$version did not exit successfully" 
            exit 1
        fi
    done
    shift

    for version in $@
    do
        ./bin/nbody-compare data/baseline-$nb_iter.nbody data/$version-$nb_iter.nbody
        if [[ "$?" = "0" ]]
        then
            echo -e "$version .. ${green}OK$reset"
        else
            echo -e "$version .. ${red}NOK$reset"
            pass=false
        fi
    done

    if [[ $pass = "true" ]]
    then
        echo -e "-> ${green}all tests passed successfully$reset"
    else
        echo -e "-> ${red}tests failed$reset"
        exit 1
    fi
}

test $versions

