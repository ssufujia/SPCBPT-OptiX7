#pragma once

#include <stdexcept>
#include <string>

namespace spcbpt::detail
{

class SamplingProgressGuard
{
  public:
    explicit SamplingProgressGuard( int max_no_progress_iterations = 64 )
        : m_max_no_progress_iterations( max_no_progress_iterations )
    {
    }

    void record( int added_samples, const char* operation )
    {
        if( added_samples > 0 )
        {
            m_no_progress_iterations = 0;
            return;
        }
        if( ++m_no_progress_iterations >= m_max_no_progress_iterations )
        {
            throw std::runtime_error(
                std::string( operation )
                + " could not collect valid samples after "
                + std::to_string( m_max_no_progress_iterations )
                + " consecutive trace batches"
            );
        }
    }

  private:
    int m_max_no_progress_iterations;
    int m_no_progress_iterations = 0;
};

} // namespace spcbpt::detail
