#include <mpi.h>
#include <torch/extension.h>
#include <stdexcept>
#include <iostream>
#include <tuple>

static bool mpi_initialized = false;

void ensure_mpi_initialized() {
    if (!mpi_initialized) {
        MPI_Init(nullptr, nullptr);
        mpi_initialized = true;
    }
}

inline std::tuple<MPI_Datatype, size_t> get_typesize(torch::ScalarType dtype) {
    switch (dtype) {
        case torch::kFloat32:
            return std::make_tuple(MPI_FLOAT, sizeof(float));
        case torch::kFloat64:
            return std::make_tuple(MPI_DOUBLE, sizeof(double));
        case torch::kInt32:
            return std::make_tuple(MPI_INT, sizeof(int32_t));
        case torch::kInt64:
            return std::make_tuple(MPI_LONG_LONG, sizeof(int64_t));
        case torch::kUInt8:
            return std::make_tuple(MPI_UNSIGNED_CHAR, sizeof(uint8_t));
        case torch::kInt8:
            return std::make_tuple(MPI_CHAR, sizeof(int8_t));
        // Add more types as needed
        default:
            throw std::runtime_error("Unsupported torch::ScalarType for MPI communication");
    }
}


MPI_Datatype get_mpi_datatype(const torch::Tensor& tensor) {
    switch (tensor.scalar_type()) {
        case torch::kFloat32: return MPI_FLOAT;
        case torch::kFloat64: return MPI_DOUBLE;
        case torch::kInt32: return MPI_INT;
        case torch::kInt64: return MPI_LONG_LONG;
        case torch::kUInt8: return MPI_UNSIGNED_CHAR;
        case torch::kInt8: return MPI_CHAR;
	// Add more cases as needed
	default: throw std::runtime_error("Unsupported tensor dtype: " + std::string(torch::toString(tensor.scalar_type())));
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

void mpi_allreduce(torch::Tensor &tensor) {
    ensure_mpi_initialized();

    if (!tensor.is_contiguous()) {
        tensor = tensor.contiguous();
    }

    // Get appropriate MPI datatype
    MPI_Datatype datatype = get_mpi_datatype(tensor);

    int mpi_result = MPI_Allreduce(
        MPI_IN_PLACE,
        tensor.data_ptr(),
        tensor.numel(),
        datatype,
        MPI_SUM,
        MPI_COMM_WORLD
    );

    // Check for errors
    if (mpi_result != MPI_SUCCESS) {
        throw std::runtime_error("MPI_Allreduce failed.");
    }

}

void mpi_allgather(torch::Tensor& sendbuf, torch::Tensor& recvbuf) {
    ensure_mpi_initialized();
    int count = sendbuf.numel();

    int rank = get_rank();
    int world_size = get_size();

    recvbuf.slice(0, rank * count, (rank + 1) * count).copy_(sendbuf);

    auto [datatype, typesize] = get_typesize(sendbuf.scalar_type());

    for (int i = 0; i < world_size; ++i) {
        if (i == rank) continue;
        MPI_Request reqs[2];
        // Send my data to rank i
        MPI_Isend(sendbuf.data_ptr(), count, datatype, i, 0, MPI_COMM_WORLD, &reqs[0]);
        // Receive their data into my recvbuf at offset i
        void* recv_ptr = static_cast<char*>(recvbuf.data_ptr()) + i * count * typesize;
        MPI_Irecv(recv_ptr, count, datatype, i, 0, MPI_COMM_WORLD, &reqs[1]);
        MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
    }
}

void mpi_reduce_scatter(torch::Tensor& sendbuf, torch::Tensor& recvbuf) {
    ensure_mpi_initialized();

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int count = recvbuf.numel();
    std::vector<int> recvcounts(world_size, count);

    auto [mpi_dtype, typesize] = get_typesize(sendbuf.scalar_type());

    MPI_Reduce_scatter(
        sendbuf.data_ptr(),
        recvbuf.data_ptr(),
        recvcounts.data(),
        mpi_dtype,
        MPI_SUM,
        MPI_COMM_WORLD
    );
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
