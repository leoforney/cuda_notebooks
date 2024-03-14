#include <iostream>
#include <boost/program_options.hpp>
#include "vector-operations/VectorOps.cuh"
#include "image-processing/ImageProcessing.cuh"
#include "list-sort/ListSort.cuh"
#include "audio-visualizer/AudioVisualizer.cuh"

namespace po = boost::program_options;

int main(int argc, const char* argv[]) {

    VectorOps vo = VectorOps();
    ImageProcessing ip = ImageProcessing();
    ListSort ls = ListSort();
    AudioVisualizer av = AudioVisualizer();

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message");

    vo.addParams(&desc);
    ip.addParams(&desc);
    ls.addParams(&desc);
    av.addParams(&desc);

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 0;
    }

    if (argc > 1) {
        std::string arg = argv[1];
        if (arg == "vo") {
            return vo.main(vm);
        } else if (arg == "ip") {
            return ip.main(vm);
        } else if (arg == "ls") {
            return ls.main(vm);
        } else if (arg == "av") {
            return av.main(vm);
        }
    }

    std::cout << "Usage: ./cuda-notebooks <program name>" << std::endl;
    return 0;
}