#include <mpi.h>
#include <torch/extension.h>

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("allgather", &mpi_allgather, "MPI AllGather");
    m.def("reduce_scatter", &mpi_reduce_scatter, "MPI ReduceScatter");
}