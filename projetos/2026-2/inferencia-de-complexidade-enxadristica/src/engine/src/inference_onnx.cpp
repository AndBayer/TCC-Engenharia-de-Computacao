#include "inference.hpp"

#include <onnxruntime_cxx_api.h>

#include <array>
#include <iostream>
#include <string>

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace engine
{

    namespace
    {

        // rede pelo ONNX Runtime, com entrada e saídas na CPU; tenta o provedor CUDA e, se falhar, usa a CPU
        class OnnxEvaluator final : public NetEvaluator
        {
            Ort::Env env_;
            Ort::Session session_{nullptr};
            Ort::MemoryInfo cpuMemory_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        public:
            OnnxEvaluator(const std::string &modelPath, bool tryCuda) : env_(ORT_LOGGING_LEVEL_WARNING, "engine")
            {
                // sem GPU NVIDIA o provedor CUDA pode registrar sem erro e falhar só ao criar a sessão;
                // nesse caso a sessão é refeita na CPU
                if (tryCuda)
                {
                    try
                    {
                        session_ = createSession(modelPath, true);
                        return;
                    }
                    catch (const Ort::Exception &e)
                    {
                        std::cerr << "info string warning: CUDA execution provider unavailable (" << e.what() << "); falling back to CPU" << std::endl;
                    }
                }
                session_ = createSession(modelPath, false);
            }

            // sessão do ONNX Runtime no provedor CUDA ou na CPU
            Ort::Session createSession(const std::string &modelPath, bool useCuda)
            {
                Ort::SessionOptions opts;
                opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
                if (useCuda)
                {
                    opts.SetIntraOpNumThreads(1);
                    OrtCUDAProviderOptions cuda;
                    cuda.device_id = 0;
                    opts.AppendExecutionProvider_CUDA(cuda);
                }
                else
                    opts.SetIntraOpNumThreads(0); // todos os núcleos

                // no Windows o ONNX Runtime recebe o caminho em wchar_t
#ifdef _WIN32
                return Ort::Session(env_, std::wstring(modelPath.begin(), modelPath.end()).c_str(), opts);
#else
                return Ort::Session(env_, modelPath.c_str(), opts);
#endif
            }

            // lote já codificado na CPU --> política e valor de cada posição; com o provedor CUDA, o ONNX Runtime copia os dados para a GPU e de volta
            void evaluateBatch(const float *inputs, int n, float *policies, float *values) override
            {
                std::array<int64_t, 4> shape = {n, encoding::CHANNELS, 8, 8};
                auto tensor = Ort::Value::CreateTensor<float>(cpuMemory_, const_cast<float *>(inputs), static_cast<size_t>(n) * encoding::INPUT_SIZE, shape.data(), shape.size());

                const char *inNames[] = {"board"};
                const char *outNames[] = {"policy", "value"};

                auto outputs = session_.Run(Ort::RunOptions{nullptr}, inNames, &tensor, 1, outNames, 2);
                const float *policyBatch = outputs[0].GetTensorData<float>();
                const float *valueBatch = outputs[1].GetTensorData<float>();

                std::copy(policyBatch, policyBatch + static_cast<size_t>(n) * encoding::POLICY_SIZE, policies);
                std::copy(valueBatch, valueBatch + n, values);
            }
        };

    } // namespace

    std::unique_ptr<NetEvaluator> makeOnnxEvaluator(const std::string &modelPath, bool tryCuda)
    {
        return std::make_unique<OnnxEvaluator>(modelPath, tryCuda);
    }

    // no Windows, confere se o provedor CUDA do ONNX Runtime e o driver da NVIDIA existem; fora do Windows retorna true
    bool cudaLooksAvailable(const std::string &exeDir, std::string &reason)
    {
#ifdef _WIN32
        // o ONNX Runtime procura o provedor ao lado do onnxruntime.dll (pasta do executável)
        char runtime[MAX_PATH] = "";
        if (HMODULE module = GetModuleHandleA("onnxruntime.dll"))
            GetModuleFileNameA(module, runtime, MAX_PATH);
        std::string folder = runtime[0] ? std::string(runtime) : exeDir + "\\onnxruntime.dll";
        folder = folder.substr(0, folder.find_last_of("\\/") + 1);
        const std::string provider = folder + "onnxruntime_providers_cuda.dll";
        if (GetFileAttributesA(provider.c_str()) == INVALID_FILE_ATTRIBUTES)
        {
            reason = "onnxruntime_providers_cuda.dll not found";
            return false;
        }
        // só verifica se o driver existe; carregar como dados não executa o driver
        HMODULE driver = LoadLibraryExA("nvcuda.dll", nullptr, LOAD_LIBRARY_AS_DATAFILE | LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (!driver)
        {
            reason = "NVIDIA driver (nvcuda.dll) not found";
            return false;
        }
        FreeLibrary(driver);
#else
        (void)exeDir;
        (void)reason;
#endif
        return true;
    }

} // namespace engine
