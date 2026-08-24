#include <api/api_pyausaxs.h>

#include <cstddef>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

extern "C" void deallocate(int object_id, int* status);

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: probe_rigidbody_api <script>\n";
        return 2;
    }

    std::ifstream input(argv[1]);
    std::string script{
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()
    };
    if (!input && script.empty()) {
        std::cerr << "failed to read script: " << argv[1] << '\n';
        return 2;
    }

    int status = 0;
    int rigidbody_id = rigidbody_load_script(script.c_str(), &status);
    std::cout << "load: id=" << rigidbody_id << " status=" << status << '\n';
    if (status != 0) {
        return 1;
    }

    double* q = nullptr;
    double* intensity = nullptr;
    double* errors = nullptr;
    double* interpolated = nullptr;
    int n_points = 0;
    int data_id = rigidbody_run(
        rigidbody_id,
        &q,
        &intensity,
        &errors,
        &interpolated,
        &n_points,
        &status
    );

    std::cout << "run: data_id=" << data_id
              << " status=" << status
              << " n_points=" << n_points << '\n';
    std::cout << "pointers: q=" << static_cast<void*>(q)
              << " I=" << static_cast<void*>(intensity)
              << " I_err=" << static_cast<void*>(errors)
              << " I_interp=" << static_cast<void*>(interpolated) << '\n';
    if (status == 0 && n_points > 0) {
        std::cout << "first: q=" << q[0]
                  << " I=" << intensity[0]
                  << " I_err=" << errors[0]
                  << " I_interp=" << interpolated[0] << '\n';
    }

    int deallocate_status = 0;
    deallocate(data_id, &deallocate_status);
    std::cout << "deallocate data: status=" << deallocate_status << '\n';
    deallocate(rigidbody_id, &deallocate_status);
    std::cout << "deallocate rigidbody: status=" << deallocate_status << '\n';
    return status == 0 ? 0 : 1;
}
