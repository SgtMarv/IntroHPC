#! /usr/bin/ruby

lines = File.open(ARGV[0]).readlines

(0..10).each do |i|
    avg = 0.0
    lines.each do |line|
        avg += line.split[i].to_f
    end
    avg = (avg/lines.count)

    err = 0.0
    lines.each do |line|
        err += (line.split[i].to_f-avg)**2
    end

    err = Math.sqrt((1/(lines.count-1.0))*err)

    puts  (2**i).to_s + " "+ avg.to_s + " " + err.to_s + "\n"
end
