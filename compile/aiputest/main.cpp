// Copyright (C) 2022-2023 siengine All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0


/**
 * @file  main.cpp
 * @brief AIPU UMD test application: basic profiler test for arm64 platforms
 */

#include <stdio.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <iostream>
#include <string.h>
#include <errno.h>
#include <vector>
#include <fstream>
#include <math.h>
#include <algorithm>
#include "standard_api.h"

#define AIPU_ERR() printf
#define AIPU_INFO() printf

int dump_perfdata(aipu_ctx_handle_t *m_ctx, uint64_t graph_id, uint64_t job_id)
{
    aipu_status_t sts = AIPU_STATUS_SUCCESS;
    const char* msg = nullptr;
    aipu_tensor_desc_t desc;
    std::string perfdata_fname;
    uint32_t cnt;
    int ret = 0;

    sts = aipu_get_tensor_count(m_ctx, graph_id, AIPU_TENSOR_TYPE_PROFILER, &cnt);
    if (sts != AIPU_STATUS_SUCCESS) {
        aipu_get_error_message(m_ctx, sts, &msg);
        AIPU_ERR()("aipu_get_tensor_descriptor(%d): %s\n",
            AIPU_TENSOR_TYPE_PROFILER, msg);
        ret = -1;
        return ret;
    } else if (cnt == 0) {
        // printf("No profiler data\n");
        ret = -1;
        return ret;
    }

    sts = aipu_get_tensor_descriptor(m_ctx, graph_id, AIPU_TENSOR_TYPE_PROFILER, 0, &desc);
    if (sts != AIPU_STATUS_SUCCESS) {
        aipu_get_error_message(m_ctx, sts, &msg);
        AIPU_ERR()("aipu_get_tensor_descriptor(%d): %s\n",
            AIPU_TENSOR_TYPE_PROFILER, msg);
        ret = -1;
        return ret;
    }

    perfdata_fname = "./PerfData.bin";
    AIPU_INFO()("perfdata file: %s\n", perfdata_fname.c_str());

    std::ofstream ofs(perfdata_fname, std::ios::binary);
    if (!ofs.is_open()) {
        AIPU_ERR()("open: %s [fail]\n", perfdata_fname.c_str());
        ret = -1;
        return ret;
    }

    char*buffer = new char[desc.size];
    sts = aipu_get_tensor(m_ctx, job_id, AIPU_TENSOR_TYPE_PROFILER, 0, buffer);
    if (sts != AIPU_STATUS_SUCCESS) {
        aipu_get_error_message(m_ctx, sts, &msg);
        AIPU_ERR()("get profiler tensor: %s [fail]\n", msg);
        ret = -1;
        goto finish;
    }
    AIPU_INFO()("get profiler tensor success");

    ofs.write(buffer, desc.size);

finish:

    delete []buffer;
    ofs.close();
    return ret;
}

template<typename T>
bool prob_comparator(const std::pair<int,T>& a, const std::pair<int,T>& b) {
    return a.second > b.second;
}
template<typename T>
int post_process(void* output_data, aipu_tensor_desc_t& output_desc)
{
    const T* output = (const T*)output_data;
    std::vector<std::pair<int, T> > tempres;
    int nclass =1001;
    for (int n = 0; n < nclass; ++n){
        //PRN_INF("--%d %d\t",n,output[n]);
        tempres.push_back(std::make_pair(n, output[n]));
    }

    std::sort(tempres.begin(), tempres.end(), prob_comparator<T>);

    printf("output_desc zero_point: %.4f scale: %.4f\n",output_desc.zero_point, output_desc.scale);
    //De-Quantization: convert to float
	//data_float = (data_fixed + zerop) / scale
	int top5 = 5;
    for (int n = 0; n < top5; ++n){
        printf("idx: %d  fval: %.4f\n", tempres[n].first,(tempres[n].second + output_desc.zero_point) / output_desc.scale );
    }
    return 0;
}

int load_file_helper(const char* fname, char** dest)
{

    if ((nullptr == fname) || (nullptr == dest) ){
        return -1;
    }

    *dest = nullptr;
    struct stat finfo;
    if (stat(fname, &finfo) != 0){
        AIPU_ERR()("open file failed: %s! (errno = %d)\n", fname, errno);
        return -1;
    }

    int fd = open(fname, O_RDONLY);
    if (fd <= 0){
        AIPU_ERR()("open file failed: %s! (errno = %d)\n", fname, errno);
        return -1;
    }

    *dest = new char[finfo.st_size];

    if (read(fd, *dest, finfo.st_size) < 0)
    {
        AIPU_ERR()("load file failed: %s! (errno = %d)\n", fname, errno);
        close(fd);
        delete[] *dest;
        *dest = nullptr;
        return -1;
    }
    close(fd);
    return finfo.st_size;
}

int main(int argc, char* argv[])
{
    aipu_status_t ret = AIPU_STATUS_SUCCESS;
    aipu_create_job_cfg_t create_job_cfg = {0};
    aipu_ctx_handle_t* ctx;
    const char* msg = nullptr;
    uint64_t graph_id, job_id;
    uint32_t input_cnt, output_cnt;
    std::vector<aipu_tensor_desc_t> input_desc;
    char* input_data = nullptr;
    int input_data_sz = 0;

    std::vector<aipu_tensor_desc_t> output_desc;
    std::vector<char*> output_data;

    uint32_t frame_cnt = 1;
    int pass = -1;

    // printf("usage: ./aipu_test aipu.bin input0.bin \n");

    const char* bin_file_name = argv[1] ;
    const char* fname = argv[2] ;
    ret = aipu_init_context(&ctx);
    if (ret != AIPU_STATUS_SUCCESS)
    {
        aipu_get_error_message(ctx, ret, &msg);
        AIPU_ERR()("aipu_init_context: %s\n", msg);
        goto finish;
    }
    // AIPU_INFO()("aipu_init_context success\n");


    ret = aipu_load_graph(ctx, bin_file_name, &graph_id);
    if (ret != AIPU_STATUS_SUCCESS)
    {
        aipu_get_error_message(ctx, ret, &msg);
        AIPU_ERR()("aipu_load_graph_helper: %s (%s)\n",
            msg, bin_file_name);
        goto deinit_ctx;
    }
    // AIPU_INFO()("aipu_load_graph_helper success: %s\n", bin_file_name);

    ret = aipu_get_tensor_count(ctx, graph_id, AIPU_TENSOR_TYPE_INPUT, &input_cnt);
    if (ret != AIPU_STATUS_SUCCESS)
    {
        aipu_get_error_message(ctx, ret, &msg);
        AIPU_ERR()("aipu_get_tensor_count: %s\n", msg);
        goto unload_graph;
    }
    //AIPU_INFO()("aipu_get_tensor_count success: input cnt = %d\n", input_cnt);

    for (uint32_t i = 0; i < input_cnt; i++)
    {
        aipu_tensor_desc_t desc;
        ret = aipu_get_tensor_descriptor(ctx, graph_id, AIPU_TENSOR_TYPE_INPUT, i, &desc);
        if (ret != AIPU_STATUS_SUCCESS)
        {
            aipu_get_error_message(ctx, ret, &msg);
            AIPU_ERR()("aipu_get_tensor_descriptor: %s\n", msg);
            goto unload_graph;
        }
        input_desc.push_back(desc);
    }

    //load input data
    input_data_sz = load_file_helper(fname, &input_data);
    if(input_data_sz <= 0 ){
       printf("load_file_helper <0 ");
       goto unload_graph;
    }

    ret = aipu_get_tensor_count(ctx, graph_id, AIPU_TENSOR_TYPE_OUTPUT, &output_cnt);
    if (ret != AIPU_STATUS_SUCCESS)
    {
        aipu_get_error_message(ctx, ret, &msg);
        AIPU_ERR()("aipu_get_tensor_count: %s\n", msg);
        goto unload_graph;
    }
    //AIPU_INFO()("aipu_get_tensor_count success: output cnt = %d\n", output_cnt);

    for (uint32_t i = 0; i < output_cnt; i++)
    {
        aipu_tensor_desc_t desc;
        ret = aipu_get_tensor_descriptor(ctx, graph_id, AIPU_TENSOR_TYPE_OUTPUT, i, &desc);
        if (ret != AIPU_STATUS_SUCCESS)
        {
            aipu_get_error_message(ctx, ret, &msg);
            AIPU_ERR()("aipu_get_tensor_descriptor: %s\n", msg);
            goto unload_graph;
        }
        output_desc.push_back(desc);
    }
    //fprintf(stderr, "[TEST INFO] aipu_get_tensor_descriptor done\n");

    ret = aipu_create_job(ctx, graph_id, &job_id, &create_job_cfg);
    if (ret != AIPU_STATUS_SUCCESS)
    {
        aipu_get_error_message(ctx, ret, &msg);
        AIPU_ERR()("aipu_create_job: %s\n", msg);
        goto unload_graph;
    }
    // AIPU_INFO()("aipu_create_job success\n");

    for (uint32_t i = 0; i < output_cnt; i++)
    {
        char* output = new char[output_desc[i].size];
        output_data.push_back(output);
    }

    /* run with with multiple frames */

    for (uint32_t frame = 0; frame < frame_cnt; frame++)
    {
        AIPU_INFO()("Frame #%u\n", frame);

        if (input_desc[0].size > input_data_sz)
        {
            AIPU_ERR()("input file %s len 0x%x < input tensor %u size 0x%x\n",
                fname, input_data_sz, 0, input_desc[0].size);
            goto clean_job;
        }
        ret = aipu_load_tensor(ctx, job_id, 0, input_data);
        if (ret != AIPU_STATUS_SUCCESS)
        {
            aipu_get_error_message(ctx, ret, &msg);
            AIPU_ERR()("aipu_load_tensor: %s\n", msg);
            goto clean_job;
        }

        ret = aipu_finish_job(ctx, job_id, -1);
        if (ret != AIPU_STATUS_SUCCESS)
        {
            aipu_get_error_message(ctx, ret, &msg);
            AIPU_ERR()("aipu_finish_job: %s\n", msg);
            goto clean_job;
        }
        // AIPU_INFO()("aipu_finish_job success\n");

        dump_perfdata(ctx, graph_id, job_id);

        for (uint32_t i = 0; i < output_cnt; i++)
        {
            ret = aipu_get_tensor(ctx, job_id, AIPU_TENSOR_TYPE_OUTPUT, i, output_data[i]);
            if (ret != AIPU_STATUS_SUCCESS)
            {
                aipu_get_error_message(ctx, ret, &msg);
                AIPU_ERR()("aipu_get_tensor: %s\n", msg);
                goto clean_job;
            }
            AIPU_INFO()("get output tensor %u success (%u/%u)\n",
                i, i+1, output_cnt);

            std::string output_filename = "output.bin." + std::to_string(i);
            std::ofstream ofs(output_filename, std::ios::binary);
            if (!ofs.is_open()) {
                AIPU_ERR()("Failed to open file %s for writing output data\n", output_filename.c_str());
                goto clean_job;
            }
            ofs.write(output_data[i], output_desc[i].size);
            ofs.close();
            // AIPU_INFO()("Saved output tensor %u to %s\n", i, output_filename.c_str());
        }
        // post_process<signed char>((void*)(output_data[0]),output_desc[0]);
    }

clean_job:
    ret = aipu_clean_job(ctx, job_id);
    if (ret != AIPU_STATUS_SUCCESS)
    {
        aipu_get_error_message(ctx, ret, &msg);
        AIPU_ERR()("aipu_clean_job: %s\n", msg);
        goto finish;
    }
    // AIPU_INFO()("aipu_clean_job success\n");

unload_graph:
    ret = aipu_unload_graph(ctx, graph_id);
    if (ret != AIPU_STATUS_SUCCESS)
    {
        aipu_get_error_message(ctx, ret, &msg);
        AIPU_ERR()("aipu_unload_graph: %s\n", msg);
        goto finish;
    }
    // AIPU_INFO()("aipu_unload_graph success\n");

deinit_ctx:
    ret = aipu_deinit_context(ctx);
    if (ret != AIPU_STATUS_SUCCESS)
    {
        aipu_get_error_message(ctx, ret, &msg);
        AIPU_ERR()("aipu_deinit_ctx: %s\n", msg);
        goto finish;
    }
    // AIPU_INFO()("aipu_deinit_ctx success\n");

finish:
    if (AIPU_STATUS_SUCCESS != ret)
        pass = -1;
    if(input_data != nullptr )
        delete [] input_data;

    for (uint32_t i = 0; i < output_data.size(); i++)
        delete[] output_data[i];

    return pass;
}