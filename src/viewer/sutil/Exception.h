#pragma once

#include <renderer/Exception.h>
#include <glad/gl.h>

#define GL_CHECK( call )                                                       \
    do                                                                         \
    {                                                                          \
        call;                                                                  \
        ::sutil::glCheck( #call, __FILE__, __LINE__ );                         \
    } while( false )

#define GL_CHECK_ERRORS() ::sutil::glCheckErrors( __FILE__, __LINE__ )

namespace sutil
{

inline const char* getGLErrorString( GLenum error )
{
    switch( error )
    {
        case GL_NO_ERROR:          return "No error";
        case GL_INVALID_ENUM:      return "Invalid enum";
        case GL_INVALID_VALUE:     return "Invalid value";
        case GL_INVALID_OPERATION: return "Invalid operation";
        case GL_OUT_OF_MEMORY:     return "Out of memory";
        default:                   return "Unknown GL error";
    }
}

inline void glCheck( const char* call, const char* file, unsigned int line )
{
    const GLenum error = glGetError();
    if( error == GL_NO_ERROR )
        return;

    std::stringstream message;
    message << "GL error " << getGLErrorString( error ) << " at "
            << file << '(' << line << "): " << call << '\n';
    throw Exception( message.str().c_str() );
}

inline void glCheckErrors( const char* file, unsigned int line )
{
    glCheck( "glGetError", file, line );
}

inline void checkGLError()
{
    const GLenum error = glGetError();
    if( error != GL_NO_ERROR )
        throw Exception( getGLErrorString( error ) );
}

} // namespace sutil
