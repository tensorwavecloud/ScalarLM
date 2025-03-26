#include <mpi.h>
#include <torch/extension.h>
#include <stdexcept>
#include <iostream>

MPI_Datatype get_mpi_datatype(const torch::Tensor& tensor) {
    switch (tensor.scalar_type()) {
        case torch::kFloat32: return MPI_FLOAT;
        case torch::kFloat64: return MPI_DOUBLE;
        case torch::kInt32: return MPI_INT;
        case torch::kInt64: return MPI_LONG_LONG;
        case torch::kUInt8: return MPI_UNSIGNED_CHAR;
        // Add more cases as needed
        default: throw std::runtime_error("Unsupported tensor dtype");
    }
}

void mpi_allgather(torch::Tensor& sendbuf, torch::Tensor& recvbuf) {
    void* send_ptr = sendbuf.data_ptr();
    void* recv_ptr = recvbuf.data_ptr();

    int count = sendbuf.numel();
    MPI_Datatype datatype = get_mpi_datatype(sendbuf);

    MPI_Allgather(send_ptr, count, datatype, recv_ptr, count, datatype, MPI_COMM_WORLD);
}

void mpi_reduce_scatter(torch::Tensor& sendbuf, torch::Tensor& recvbuf) {
    void* send_ptr = sendbuf.data_ptr();
    void* recv_ptr = recvbuf.data_ptr();
    
    MPI_Datatype datatype = get_mpi_datatype(sendbuf);
    
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    std::vector<int> recvcounts(size);
    int recv_elements = recvbuf.numel();
    for (int i = 0; i < size; ++i) {
        recvcounts[i] = recv_elements;
    }
    
    MPI_Reduce_scatter(send_ptr, recv_ptr, recvcounts.data(), datatype, MPI_SUM, MPI_COMM_WORLD);
}

void mpi_send(torch::Tensor& tensor, int dest) {
    void* ptr = tensor.data_ptr();
    int count = tensor.numel();
    MPI_Datatype datatype = get_mpi_datatype(tensor);

    MPI_Send(ptr, count, datatype, dest, 0, MPI_COMM_WORLD);
}

void mpi_recv(torch::Tensor& tensor, int source) {
    void* ptr = tensor.data_ptr();
    int count = tensor.numel();
    MPI_Datatype datatype = get_mpi_datatype(tensor);

    MPI_Status status;

    MPI_Recv(ptr, count, datatype, source, 0, MPI_COMM_WORLD, &status);

    int recv_count;
    MPI_Get_count(&status, datatype, &recv_count);

    if (recv_count != count) {
        std::cout << "Received unexpected number of elements: " << recv_count << " != " << count << std::endl;
        throw std::runtime_error("Received unexpected number of elements: " + std::to_string(recv_count) + " != " + std::to_string(count));
    }

    if (status.MPI_SOURCE != source) {
        std::cout << "Received message from unexpected source: " << status.MPI_SOURCE << " != " << source << std::endl;
        throw std::runtime_error("Received message from unexpected source: " + std::to_string(status.MPI_SOURCE) + " != " + std::to_string(source));
    }

    // Check for errors
    int error;

    MPI_Error_class(status.MPI_ERROR, &error);

    if (error != MPI_SUCCESS) {
        char error_string[MPI_MAX_ERROR_STRING] = {0};
        int length;
        MPI_Error_string(error, error_string, &length);
        std::cout << "Received error: " << error_string << " (" << error << ")" << std::endl;
        throw std::runtime_error(error_string);
    }
}

void barrier() {
    MPI_Barrier(MPI_COMM_WORLD);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("allgather", &mpi_allgather, "MPI AllGather");
    m.def("reduce_scatter", &mpi_reduce_scatter, "MPI ReduceScatter");
    m.def("send", &mpi_send, "MPI Send");
    m.def("recv", &mpi_recv, "MPI Recv");
    m.def("barrier", &barrier, "MPI Barrier");
}