#include <iostream>
#include <boost/program_options.hpp>
#include "vector-operations/VectorOps.cuh"
#include "image-processing/ImageProcessing.cuh"

namespace po = boost::program_options;

int main(int argc, const char* argv[]) {

    VectorOps vo = VectorOps();
    ImageProcessing ip = ImageProcessing();

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message");

    vo.addParams(&desc);
    ip.addParams(&desc);

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
        }
    }

    std::cout << "Usage: ./cuda-notebooks <program name>" << std::endl;
    return 0;
}