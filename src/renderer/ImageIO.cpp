#include <renderer/ImageIO.h>

#include <renderer/Exception.h>
#include <tinygltf/stb_image_write.h>

namespace spcbpt
{

void saveRgba8Png(
    const std::string& path,
    const uchar4* pixels,
    unsigned int width,
    unsigned int height
)
{
    if( path.empty() )
        throw sutil::Exception( "PNG output path is empty" );
    if( !pixels || width == 0 || height == 0 )
        throw sutil::Exception( "PNG output has invalid pixels or dimensions" );

    const int result = stbi_write_png(
        path.c_str(),
        static_cast<int>( width ),
        static_cast<int>( height ),
        4,
        pixels,
        static_cast<int>( width * sizeof( uchar4 ) )
    );
    if( result == 0 )
        throw sutil::Exception( ( "Failed to write PNG: " + path ).c_str() );
}

} // namespace spcbpt
