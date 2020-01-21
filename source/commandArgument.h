#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace sis {

class CommandArgument {
public:
    CommandArgument(int argc, char* argv[]);

    const std::string find(const std::string& key, 
                           const std::string& defaultValue = "") const;

    bool isHelpMessageRequested() const;
    void printHelpMessage() const;

private:
    bool _isHelpMessageRequested;

    std::unordered_map<std::string, std::string> _arguments;
};

} // namespace sis