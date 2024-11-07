# Hackathon Notes

Used this docker command
```
docker run \
    -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
    -v $HOME/jordan_hackin/computer_use_demo/container/.cache:/home/computeruse/.cache \
    -v $HOME/jordan_hackin/computer_use_demo/container/.mozilla:/home/computeruse/.mozilla \
    -v $HOME/jordan_hackin/computer_use_demo/anthropic-quickstarts/computer-use-demo/anthropic_config:/home/computeruse/.anthropic \
    -v $HOME/jordan_hackin/computer_use_demo/anthropic-quickstarts/computer-use-demo/computer_use_demo:/home/computeruse/computer_use_demo \
    -v $HOME/jordan_hackin/computer_use_demo/anthropic-quickstarts/computer-use-demo/image:/home/computeruse/custom_image \
    -p 5900:5900 \
    -p 8501:8501 \
    -p 6080:6080 \
    -p 8080:8080 \
    --entrypoint "custom_image/entrypoint.sh" \
    -it ghcr.io/anthropics/anthropic-quickstarts:computer-use-demo-latest

```

Mouse highlighter: vokoscreen
Slides: https://docs.google.com/presentation/d/1UmBM9XTN9kzr9fvSJJi6i82yCx44mK3ryXoUbjnHwqs/edit#slide=id.g3127d6e3999_0_139


# Anthropic Quickstarts

Anthropic Quickstarts is a collection of projects designed to help developers quickly get started with building  applications using the Anthropic API. Each quickstart provides a foundation that you can easily build upon and customize for your specific needs.

## Getting Started

To use these quickstarts, you'll need an Anthropic API key. If you don't have one yet, you can sign up for free at [console.anthropic.com](https://console.anthropic.com).

## Available Quickstarts

### Customer Support Agent

A customer support agent powered by Claude. This project demonstrates how to leverage Claude's natural language understanding and generation capabilities to create an AI-assisted customer support system with access to a knowledge base.

[Go to Customer Support Agent Quickstart](./customer-support-agent)

### Financial Data Analyst

A financial data analyst powered by Claude. This project demonstrates how to leverage Claude's capabilities with interactive data visualization to analyze financial data via chat.

[Go to Financial Data Analyst Quickstart](./financial-data-analyst)

### Computer Use Demo

An environment and tools that Claude can use to control a desktop computer. This project demonstrates how to leverage the computer use capabilities of the the new Claude 3.5 Sonnet model.

[Go to Computer Use Demo Quickstart](./computer-use-demo)

## General Usage

Each quickstart project comes with its own README and setup instructions. Generally, you'll follow these steps:

1. Clone this repository
2. Navigate to the specific quickstart directory
3. Install the required dependencies
4. Set up your Anthropic API key as an environment variable
5. Run the quickstart application

## Explore Further

To deepen your understanding of working with Claude and the Anthropic API, check out these resources:

- [Anthropic API Documentation](https://docs.anthropic.com)
- [Anthropic Cookbook](https://github.com/anthropics/anthropic-cookbook) - A collection of code snippets and guides for common tasks
- [Anthropic API Fundamentals Course](https://github.com/anthropics/courses/tree/master/anthropic_api_fundamentals)

## Contributing

We welcome contributions to the Anthropic Quickstarts repository! If you have ideas for new quickstart projects or improvements to existing ones, please open an issue or submit a pull request.

## Community and Support

- Join our [Anthropic Discord community](https://www.anthropic.com/discord) for discussions and support
- Check out the [Anthropic support documentation](https://support.anthropic.com) for additional help

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
