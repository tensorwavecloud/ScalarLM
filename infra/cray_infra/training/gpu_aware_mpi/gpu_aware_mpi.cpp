#include <mpi.h>
#include <torch/extension.h>
#include <stdexcept>
#include <iostream>

static bool mpi_initialized = false;

void ensure_mpi_initialized() {
    if (!mpi_initialized) {
        MPI_Init(nullptr, nullptr);
        mpi_initialized = true;
    }
}

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

void mpi_allreduce(torch::Tensor &tensor) {
    ensure_mpi_initialized();
    int count = tensor.numel();
    void *data = tensor.data_ptr();
    MPI_Datatype datatype = get_mpi_datatype(tensor);

    MPI_Allreduce(MPI_IN_PLACE, data, count, datatype, MPI_SUM, MPI_COMM_WORLD);
}

void mpi_allgather(torch::Tensor& sendbuf, torch::Tensor& recvbuf) {
    ensure_mpi_initialized();
    void* send_ptr = sendbuf.data_ptr();
    void* recv_ptr = recvbuf.data_ptr();

    int count = sendbuf.numel();
    MPI_Datatype datatype = get_mpi_datatype(sendbuf);

    MPI_Allgather(send_ptr, count, datatype, recv_ptr, count, datatype, MPI_COMM_WORLD);
}

void mpi_reduce_scatter(torch::Tensor& sendbuf, torch::Tensor& recvbuf) {
    ensure_mpi_initialized();
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
    ensure_mpi_initialized();
    void* ptr = tensor.data_ptr();
    int count = tensor.numel();
    MPI_Datatype datatype = MPI_FLOAT;

    MPI_Send(ptr, count, datatype, dest, 0, MPI_COMM_WORLD);
}

void mpi_recv(torch::Tensor& tensor, int source) {
    ensure_mpi_initialized();
    void* ptr = tensor.data_ptr();
    int count = tensor.numel();
    MPI_Datatype datatype = MPI_FLOAT;

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
}

void barrier() {
    ensure_mpi_initialized();
    MPI_Barrier(MPI_COMM_WORLD);
}

int get_rank() {
    ensure_mpi_initialized();
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

int get_size() {
    ensure_mpi_initialized();
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
}

void finalize_mpi() {
    if (mpi_initialized) {
        MPI_Finalize();
        mpi_initialized = false;
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("allgather", &mpi_allgather, "MPI AllGather");
    m.def("allreduce", &mpi_allreduce, "MPI AllReduce");
    m.def("reduce_scatter", &mpi_reduce_scatter, "MPI ReduceScatter");
    m.def("send", &mpi_send, "MPI Send");
    m.def("recv", &mpi_recv, "MPI Recv");
    m.def("barrier", &barrier, "MPI Barrier");
    m.def("get_rank", &get_rank, "Get MPI rank");
    m.def("get_size", &get_size, "Get MPI world size");
    m.def("finalize_mpi", &finalize_mpi, "Finalize MPI");
}