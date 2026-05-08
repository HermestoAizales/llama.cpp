find_package(Threads REQUIRED)

set(TARGET llama-optimizer)

add_executable(${TARGET}
    main.cpp
    optimizer.cpp
    optimizer.h
    preset.h
)

install(TARGETS ${TARGET} RUNTIME)

target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/ggml/src
)

target_link_libraries(${TARGET} PRIVATE
    common
    llama
    ggml
    ${CMAKE_THREAD_LIBS_INIT}
)

target_compile_features(${TARGET} PRIVATE cxx_std_17)
